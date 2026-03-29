#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 


import time
import json
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from datetime import datetime

from config import ModelConfig, TrainingConfig, PROCESSED_DIR, MODELS_DIR
from model import build_model
from dataset import ChagasDataset
from loss import ClassBalancedFocalLoss



def cinc_challenge_score(y_true, y_prob, alpha=0.05):
    
    n = len(y_true)
    n_top = max(1, int(np.floor(alpha * n)))
    n_positive = int((y_true == 1).sum())
    if n_positive == 0:
        return 0.0

    ranked_idx = np.argsort(y_prob)[::-1]
    top_idx = ranked_idx[:n_top]

    tp_in_top = int(y_true[top_idx].sum())
    return float(tp_in_top / n_positive)


def compute_metrics(y_true, y_prob, threshold=0.5):
    
    challenge = cinc_challenge_score(y_true, y_prob, alpha=0.05)

    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.0

    preds = (y_prob >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    
    bal_acc = (sens + spec) / 2.0

    return {
        "balanced_accuracy": float(bal_acc),    
        "auroc": auroc,                         
        "challenge_score": float(challenge),    
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def find_optimal_threshold(y_true, y_prob):
    
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal = (sens + spec) / 2.0
        if bal > best_s:
            best_s, best_t = bal, t
    return round(float(best_t), 3), float(best_s)


# TRAINING

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    
    model.train()
    total_loss = 0.0
    for ecg, labels, ages, sexes in tqdm(loader, desc="  train", leave=False):
        ecg    = ecg.to(device)
        labels = labels.to(device)
        ages   = ages.to(device)
        sexes  = sexes.to(device)

        optimizer.zero_grad()
        logits = model(ecg, ages, sexes)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate on validation set; returns metrics + raw predictions."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for ecg, labels, ages, sexes in tqdm(loader, desc="  eval ", leave=False):
        ecg    = ecg.to(device)
        labels = labels.to(device)
        ages   = ages.to(device)
        sexes  = sexes.to(device)

        logits = model(ecg, ages, sexes)
        total_loss += criterion(logits, labels).item()

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.cpu().numpy().tolist())

    p = np.array(all_probs)
    l = np.array(all_labels)
    m = compute_metrics(l, p, threshold)
    m.update({"loss": total_loss / max(len(loader), 1), "probs": p, "labels": l})
    return m


# MAIN TRAINING 


def train_ensemble(tcfg=None):
    
    if tcfg is None:
        tcfg = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    output_dir = MODELS_DIR / tcfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = PROCESSED_DIR / "metadata.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {meta_path}\n"
            "Run:  python preprocess_data.py"
        )

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    all_samples = metadata["all_samples"]



    if "test" in metadata and "train" in metadata:
        
        dev_samples = metadata["train"] + metadata.get("val", [])
        n_test = len(metadata["test"])
        print(f"\n  DATA LEAKAGE PREVENTION:")
        print(f"    Development set (train+val): {len(dev_samples):,} samples  → used for 5-fold CV")
        print(f"    Held-out test set:           {n_test:,} samples  → NEVER used in training")
    else:
        dev_samples = all_samples
        print(f"  WARNING: No train/test split found in metadata.")
        print(f"    Using all {len(dev_samples):,} samples for CV (no held-out test).")

    n_pos = sum(1 for s in dev_samples if s["label"] == 1)
    n_neg = len(dev_samples) - n_pos

    print("=" * 65)
    print(f"  CHAGASVISION TRAINING — {tcfg.experiment_name}")
    print("=" * 65)
    print(f"  Samples : {len(dev_samples):,}  ({n_pos:,} positive / {n_neg:,} negative)")
    print(f"  Ratio   : 1:{n_neg // n_pos} (positive:negative)")
    print(f"  Device  : {device}")
    print(f"\n  CLASS IMBALANCE COUNTERMEASURES:")
    print(f"    ① Weighted Sampling  : {tcfg.oversample_factor}x oversample for positives")
    print(f"    ② Focal Loss         : γ={tcfg.focal_gamma}, smoothing={tcfg.label_smoothing}")
    print(f"    ③ Augmentation       : 80% positive / 20% negative augmentation rate")
    print(f"    ④ Stratified K-Fold  : {tcfg.n_folds} folds, each preserves {n_pos/len(dev_samples)*100:.1f}% positive rate")
    print(f"    ⑤ Threshold Opt.     : post-training calibration on validation predictions")
    print(f"    ⑥ Gradient Clipping  : max_norm={tcfg.gradient_clip}")


    labels = np.array([s["label"] for s in dev_samples])
    skf = StratifiedKFold(
        n_splits=tcfg.n_folds,
        shuffle=True,
        random_state=tcfg.seed,
    )

    fold_results = []
    all_val_probs, all_val_labels = [], []
    t_start = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dev_samples)), labels)):
        fold_num = fold_i + 1
        fold_dir = output_dir / f"fold_{fold_num}"
        fold_dir.mkdir(exist_ok=True)

        train_samples = [dev_samples[i] for i in train_idx]
        val_samples   = [dev_samples[i] for i in val_idx]

        n_train_pos = sum(1 for s in train_samples if s["label"] == 1)
        n_val_pos   = sum(1 for s in val_samples if s["label"] == 1)

        print(f"\n{'='*65}")
        print(f"  FOLD {fold_num}/{tcfg.n_folds}")
        print(f"  Train: {len(train_samples):,} ({n_train_pos} pos = "
              f"{100*n_train_pos/len(train_samples):.1f}%)  "
              f"Val: {len(val_samples):,} ({n_val_pos} pos)")
        print(f"{'='*65}")

        # ── Datasets ──────────────────────────────────────────────
        # TECHNIQUE ③ lives in ChagasDataset — augmentation is applied
        # on-the-fly in __getitem__, biased toward positive samples.
        data_cache = {}  # shared memory-map cache between train/val
        train_ds = ChagasDataset(
            train_samples, PROCESSED_DIR,
            training=True, augment=tcfg.use_augmentation,
            data_cache=data_cache,
        )
        val_ds = ChagasDataset(
            val_samples, PROCESSED_DIR,
            training=False, augment=False,
            data_cache=data_cache,
        )

        sample_weights = [
            tcfg.oversample_factor if s["label"] == 1 else 1.0
            for s in train_samples
        ]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=tcfg.batch_size,
            sampler=train_sampler,   # NOT shuffle — sampler handles it
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=tcfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = build_model().to(device)
        if fold_num == 1:
            print(f"  Parameters: {model.count_parameters():,}")

        
        criterion = ClassBalancedFocalLoss(
            n_pos, n_neg,
            gamma=tcfg.focal_gamma,
            smoothing=tcfg.label_smoothing,
        )
        if fold_num == 1:
            print(f"  Loss alpha (positive weight): {criterion.alpha:.4f}")

        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
        )

       
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4,
        )
        best_score = 0.0
        patience_count = 0

        for epoch in range(1, tcfg.max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion,
                optimizer, device, tcfg.gradient_clip,
            )
            val_m = evaluate(model, val_loader, criterion, device)

            
            select_metric = val_m["balanced_accuracy"]
            scheduler.step(select_metric)

            tag = ""
            if select_metric > best_score:
                best_score = select_metric
                patience_count = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "balanced_accuracy": best_score,
                }, fold_dir / "best_model.pth")
                tag = " ★ new best"
            else:
                patience_count += 1

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  E{epoch:02d}  loss={train_loss:.4f}  "
                f"sens={val_m['sensitivity']:.3f}  "
                f"spec={val_m['specificity']:.3f}  "
                f"balAcc={val_m['balanced_accuracy']:.4f}  "
                f"auroc={val_m['auroc']:.4f}  "
                f"TPR@5%={val_m['challenge_score']:.3f}  "
                f"lr={lr_now:.1e}{tag}"
            )

            
            if patience_count >= tcfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        ckpt = torch.load(fold_dir / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        final_m = evaluate(model, val_loader, criterion, device)

        fold_results.append({
            "fold": fold_num,
            "best_balanced_acc": best_score,
            "best_auroc": final_m["auroc"],
            "best_sens": final_m["sensitivity"],
            "best_spec": final_m["specificity"],
            "best_challenge": final_m["challenge_score"],
        })
        all_val_probs.extend(final_m["probs"].tolist())
        all_val_labels.extend(final_m["labels"].tolist())

        print(f"  Fold {fold_num} — BalAcc={best_score:.4f}  "
              f"AUROC={final_m['auroc']:.4f}  "
              f"Sens={final_m['sensitivity']:.3f}  "
              f"Spec={final_m['specificity']:.3f}")

    

    total_time = time.time() - t_start

    opt_thresh, _ = find_optimal_threshold(
        np.array(all_val_labels), np.array(all_val_probs),
    )

    oof_m = compute_metrics(np.array(all_val_labels), np.array(all_val_probs), opt_thresh)

    ba_scores    = [r["best_balanced_acc"] for r in fold_results]
    auroc_scores = [r["best_auroc"]        for r in fold_results]
    sens_scores  = [r["best_sens"]         for r in fold_results]
    spec_scores  = [r["best_spec"]         for r in fold_results]
    cs_scores    = [r["best_challenge"]    for r in fold_results]

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  PER-FOLD AVERAGES:")
    print(f"    Balanced Accuracy : {np.mean(ba_scores):.4f} ± {np.std(ba_scores):.4f}")
    print(f"    AUC-ROC           : {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
    print(f"    Sensitivity       : {np.mean(sens_scores):.4f} ± {np.std(sens_scores):.4f}")
    print(f"    Specificity       : {np.mean(spec_scores):.4f} ± {np.std(spec_scores):.4f}")
    print(f"    CinC TPR@5%       : {np.mean(cs_scores):.4f} ± {np.std(cs_scores):.4f}")
    print(f"\n  ENSEMBLE (combined OOF predictions, threshold={opt_thresh}):")
    print(f"    Balanced Accuracy : {oof_m['balanced_accuracy']:.4f}")
    print(f"    AUC-ROC           : {oof_m['auroc']:.4f}")
    print(f"    Sensitivity       : {oof_m['sensitivity']:.4f}  ({oof_m['tp']:,} detected / {oof_m['tp']+oof_m['fn']:,} total Chagas)")
    print(f"    Specificity       : {oof_m['specificity']:.4f}  ({oof_m['tn']:,} correct / {oof_m['tn']+oof_m['fp']:,} total healthy)")
    print(f"    Missed cases (FN) : {oof_m['fn']:,}")
    print(f"    False alarms (FP) : {oof_m['fp']:,}")
    print(f"    F1 Score          : {oof_m['f1']:.4f}")
    print(f"    CinC TPR@5%       : {oof_m['challenge_score']:.4f}")
    print(f"    Total time        : {total_time / 60:.1f} min")

    mcfg = ModelConfig()
    with open(output_dir / "ensemble_config.json", "w") as f:
        json.dump({
            "n_folds": tcfg.n_folds,
            "optimal_threshold": opt_thresh,
            "model_config": {
                "num_leads": mcfg.num_leads, "seq_length": mcfg.seq_length,
                "d_model": mcfg.d_model, "nhead": mcfg.nhead,
                "num_transformer_layers": mcfg.num_transformer_layers,
                "dim_feedforward": mcfg.dim_feedforward,
                "use_metadata": mcfg.use_metadata, "metadata_dim": mcfg.metadata_dim,
                "use_se": mcfg.use_se,
            },
        }, f, indent=2)

    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump({
            "fold_results": fold_results,
            "mean_balanced_accuracy": float(np.mean(ba_scores)),
            "std_balanced_accuracy": float(np.std(ba_scores)),
            "mean_auroc": float(np.mean(auroc_scores)),
            "std_auroc": float(np.std(auroc_scores)),
            "mean_sensitivity": float(np.mean(sens_scores)),
            "mean_specificity": float(np.mean(spec_scores)),
            "mean_challenge_score": float(np.mean(cs_scores)),
            "ensemble_metrics": oof_m,
            "optimal_threshold": opt_thresh,
            "total_time_minutes": total_time / 60,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Saved to: {output_dir}")
    return {
        "fold_results": fold_results,
        "mean_balanced_accuracy": float(np.mean(ba_scores)),
        "mean_auroc": float(np.mean(auroc_scores)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChagasVision ensemble")
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--name",   type=str, default="ensemble_v2_improved")
    args = parser.parse_args()

    cfg = TrainingConfig(
        n_folds=args.folds,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        experiment_name=args.name,
    )
    train_ensemble(cfg)