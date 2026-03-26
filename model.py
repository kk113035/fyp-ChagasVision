"""
ChagasNet — Model Architecture
===============================
Single model definition used by training, validation, XAI, and the app.

Architecture: Multi-Scale CNN → SE Attention → Transformer Encoder → MLP
Input:  ECG [B, 12, 2048], age [B], sex [B]
Output: logit [B, 1]

References
----------
[1] Szegedy et al. (2015) Going Deeper with Convolutions, CVPR.
[2] Hu et al. (2018) Squeeze-and-Excitation Networks, CVPR.
[3] Vaswani et al. (2017) Attention Is All You Need, NeurIPS.
[4] Hannun et al. (2019) Cardiologist-level arrhythmia detection, Nature Medicine.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from config import ModelConfig


# ── Building Blocks ──────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """
    Channel attention via Squeeze-and-Excitation (Hu et al., 2018).

    Squeeze:    Global average pooling  [B,C,L] → [B,C]
    Excitation: FC → ReLU → FC → Sigmoid → per-channel scale
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        scale = x.mean(dim=2)          # [B, C]
        scale = self.fc(scale)          # [B, C]
        return x * scale.unsqueeze(2)   # [B, C, L]


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Registered as a buffer so it moves with the model to GPU/CPU.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1), :]


# ── Main Model ───────────────────────────────────────────────────────────────

class ChagasNet(nn.Module):
    """
    Multi-Scale CNN + SE Attention + Transformer for Chagas ECG detection.

    Data flow
    ---------
    ECG [B,12,2048]
      → Multi-scale Conv1d (k=3,7,15)  → [B, 96, 1024]
      → BatchNorm → ReLU → MaxPool     → [B, 96, 512]
      → SE attention
      → Conv1d (k=7)                    → [B, 128, 256]
      → BatchNorm → ReLU → MaxPool     → [B, 128, 128]
      → SE attention
      → Linear projection              → [B, 128, 128]
      → Positional encoding
      → Transformer encoder (2 layers)  → [B, 128, 128]
      → Adaptive average pool           → [B, 128]
      → Concat metadata (age 4d + sex 4d) → [B, 136]
      → MLP classifier                  → [B, 1]
    """

    def __init__(self, cfg: ModelConfig = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig(**kwargs)
        self.cfg = cfg

        n_ms = len(cfg.kernel_sizes)       # 3 branches
        ms_total = cfg.ms_out_channels * n_ms  # 96

        # ── Multi-scale CNN ──────────────────────────────────────────────
        self.ms_convs = nn.ModuleList([
            nn.Conv1d(cfg.num_leads, cfg.ms_out_channels,
                      kernel_size=k, stride=2, padding=k // 2)
            for k in cfg.kernel_sizes
        ])
        self.bn1   = nn.BatchNorm1d(ms_total)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.se1   = SqueezeExcitation(ms_total, cfg.se_reduction) if cfg.use_se else nn.Identity()

        # ── Second conv block ────────────────────────────────────────────
        self.conv2 = nn.Conv1d(ms_total, cfg.conv2_out,
                               kernel_size=cfg.conv2_kernel, stride=2,
                               padding=cfg.conv2_kernel // 2)
        self.bn2   = nn.BatchNorm1d(cfg.conv2_out)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.se2   = SqueezeExcitation(cfg.conv2_out, cfg.se_reduction) if cfg.use_se else nn.Identity()

        # ── Transformer encoder ──────────────────────────────────────────
        self.proj    = nn.Linear(cfg.conv2_out, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.num_transformer_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Metadata encoder ────────────────────────────────────────────
        self.use_metadata = cfg.use_metadata
        meta_dim = 0
        if cfg.use_metadata:
            half = cfg.metadata_dim // 2
            self.age_fc  = nn.Linear(1, half)
            self.sex_emb = nn.Embedding(2, half)
            meta_dim = cfg.metadata_dim

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model + meta_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    # ── Weight initialisation ────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, ecg: torch.Tensor,
                age: torch.Tensor = None,
                sex: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            ecg: [B, 12, 2048]
            age: [B] patient age in years   (optional)
            sex: [B] 0=female, 1=male       (optional)
        Returns:
            logits: [B, 1]
        """
        # Multi-scale convolution
        branches = [conv(ecg) for conv in self.ms_convs]
        x = torch.cat(branches, dim=1)       # [B, 96, 1024]

        x = torch.relu(self.bn1(x))
        x = self.drop1(self.pool1(x))        # [B, 96, 512]
        x = self.se1(x)

        # Second conv block
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))        # [B, 128, 128]
        x = self.se2(x)

        # Transformer
        x = x.permute(0, 2, 1)               # [B, 128, 128] → [B, L, C]
        x = self.proj(x)                     # [B, L, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)              # [B, L, d_model]

        # Pool
        x = x.permute(0, 2, 1)               # [B, d_model, L]
        x = self.pool(x).squeeze(-1)         # [B, d_model]

        # Metadata fusion
        if self.use_metadata and age is not None and sex is not None:
            age_feat = torch.relu(self.age_fc(age.float().unsqueeze(-1) / 100.0))
            sex_feat = self.sex_emb(sex.long())
            x = torch.cat([x, age_feat, sex_feat], dim=1)

        return self.classifier(x)

    # ── Utilities ────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_checkpoint(path, device="cpu", inference=True):
        """Load a trained model from checkpoint file."""
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Infer config from checkpoint if stored, else use defaults
        cfg_dict = ckpt.get("model_config", {})
        if inference:
            cfg_dict["dropout"] = 0.0
        cfg = ModelConfig(**cfg_dict)

        model = ChagasNet(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        if inference:
            model.eval()
        return model


# ── Factory / aliases ────────────────────────────────────────────────────────

# Alias for backward compatibility with any code that uses "ChagasModel"
ChagasModel = ChagasNet


def build_model(cfg: ModelConfig = None, dropout_override: float = None) -> ChagasNet:
    """Create a model.  Use ``dropout_override=0.0`` for inference."""
    import copy
    if cfg is None:
        cfg = ModelConfig()
    if dropout_override is not None:
        cfg = copy.deepcopy(cfg)
        cfg.dropout = dropout_override
    return ChagasNet(cfg)


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ChagasNet()
    print(f"Parameters: {model.count_parameters():,}")

    ecg = torch.randn(4, 12, 2048)
    age = torch.tensor([45, 60, 35, 70])
    sex = torch.tensor([0, 1, 1, 0])

    logits = model(ecg, age, sex)
    probs  = torch.sigmoid(logits)
    print(f"Output shape: {logits.shape}")
    print(f"Probabilities: {probs.squeeze().tolist()}")
