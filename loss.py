#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 



import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    
    def __init__(
        self,
        n_positive: int,
        n_negative: int,
        beta: float = 0.9999,
        gamma: float = 2.0,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

       
        E_pos = (1 - beta ** n_positive) / (1 - beta)
        E_neg = (1 - beta ** n_negative) / (1 - beta)

       
        self.alpha = E_neg / (E_pos + E_neg)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        logits  = logits.squeeze(-1)
        targets = targets.float()

        
        y_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, y_smooth, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t).pow(self.gamma)

        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()

    def extra_repr(self) -> str:
        return (f"alpha={self.alpha:.4f}, gamma={self.gamma}, "
                f"smoothing={self.smoothing}")
