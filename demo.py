#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 

import json
import pickle
import random
import argparse
import numpy as np
import h5py
import torch
from pathlib import Path
from datetime import datetime

from config import PROCESSED_DIR, MODELS_DIR

# ── Output directory ─────────────────────────────────────────────────────────
DEMO_DIR = Path(PROCESSED_DIR).parent / "demo_samples_final"

# ── Category probability boundaries ─────────────────────────────────────────
CLEAR_POS_MIN   = 0.65   # clearly positive
CLEAR_NEG_MAX   = 0.35   # clearly negative
BORDER_LOW      = 0.40   # borderline lower bound
BORDER_HIGH     = 0.60   # borderline upper bound
EDGE_AGE_MIN    = 75     # older patient edge case
EDGE_AGE_MAX    = 25     # young patient edge case


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_ensemble(ensemble_name="ensemble_v2_improved"):
    """Load trained ensemble from local models directory."""
    from model import build_model
    ens_dir = MODELS_DIR / ensemble_name
    cfg_path = ens_dir / "ensemble_config.json"

    if not cfg_path.exists():
        print(f"  ERROR: No ensemble found at {ens_dir}")
        print(f"  Make sure training has completed and models are saved.")
        return None, 0.5

    with open(cfg_path) as f:
        cfg = json.load(f)
    threshold = cfg.get("optimal_threshold", 0.5)

    models = []
    for i in range(1, cfg["n_folds"] + 1):
        p = ens_dir / f"fold_{i}" / "best_model.pth"
        if not p.exists():
            continue
        m = build_model(dropout_override=0.0)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)

    print(f"  Loaded {len(models)} fold models  |  threshold = {threshold:.3f}")
    return models if models else None, threshold


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict(models, signal, age, sex, threshold):
    """
    Run full ensemble prediction on one signal.

    Returns:
        avg_prob    : mean probability across all fold models
        per_model   : list of individual model probabilities
        prediction  : 'Chagas Positive' or 'Chagas Negative'
        agreement   : fraction of models that agree with the majority vote
    """
    if models is None:
        return None, [], None, None

    ecg = torch.from_numpy(signal).unsqueeze(0).float()
    a   = torch.tensor([age]).float()
    s   = torch.tensor([sex]).long()

    per_model = [torch.sigmoid(m(ecg, a, s)).item() for m in models]
    avg       = float(np.mean(per_model))
    n_pos     = sum(1 for p in per_model if p >= threshold)
    agreement = max(n_pos, len(per_model) - n_pos) / len(per_model)

    return avg, per_model, "Chagas Positive" if avg >= threshold else "Chagas Negative", round(agreement, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE SELECTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _score_sample(s, prob, per_model, threshold, agreement):
    """
    Assign a sample to a category and compute a diversity score.
    Used to rank candidates within each category.
    """
    if prob is None:
        return None, 0.0

    margin = abs(prob - threshold)

    # Category assignment
    if prob >= CLEAR_POS_MIN and s["label"] == 1:
        category = "clear_positive"
        strength = prob  # prefer higher probability positives
    elif prob <= CLEAR_NEG_MAX and s["label"] == 0:
        category = "clear_negative"
        strength = 1 - prob  # prefer lower probability negatives
    elif BORDER_LOW <= prob <= BORDER_HIGH:
        category = "borderline"
        strength = 1 - margin  # prefer cases closest to threshold
    else:
        category = "other"
        strength = 0.5

    return category, strength


def _age_band(age):
    """Bucket age into young / middle / older for diversity tracking."""
    if age < 35:
        return "young"
    elif age < 60:
        return "middle"
    else:
        return "older"


def select_diverse_samples(candidates, n_target, seen_sources, seen_ages,
                           seen_sexes, prefer_agreement_spread=False):
    """
    Select n_target samples from candidates, maximising diversity
    across source, age band, and sex.

    Diversity scoring:
      +2 if source not yet seen
      +1 if age band not yet seen
      +1 if sex not yet seen
      +1 if agreement is split (not unanimous) — optional

    If not enough diverse candidates exist, fills remaining slots
    with the strongest remaining candidates regardless of diversity.
    """
    selected      = []
    selected_set  = set()
    local_sources = set(seen_sources)
    local_ages    = set(seen_ages)
    local_sexes   = set(seen_sexes)

    # Pass 1: diversity-weighted selection
    # Re-score every candidate against current local state and pick best
    for _ in range(n_target):
        if not candidates:
            break

        best_c, best_score = None, -1.0
        for c in candidates:
            if id(c) in selected_set:
                continue
            src = c["source"]
            ab  = _age_band(c["age"])
            sx  = "M" if c["sex"] == 1 else "F"

            div = 0
            if src not in local_sources: div += 2
            if ab  not in local_ages:    div += 1
            if sx  not in local_sexes:   div += 1
            if prefer_agreement_spread and c.get("agreement", 1.0) < 1.0:
                div += 1

            # Combined score: diversity first, then signal strength
            score = div * 10 + c["strength"]
            if score > best_score:
                best_score, best_c = score, c

        if best_c is None:
            break

        best_c["div_score"] = best_score
        selected.append(best_c)
        selected_set.add(id(best_c))
        local_sources.add(best_c["source"])
        local_ages.add(_age_band(best_c["age"]))
        local_sexes.add("M" if best_c["sex"] == 1 else "F")

    # Pass 2: if still short, fill with strongest remaining (no diversity filter)
    if len(selected) < n_target:
        remaining = sorted(
            [c for c in candidates if id(c) not in selected_set],
            key=lambda x: x["strength"], reverse=True
        )
        for c in remaining:
            if len(selected) >= n_target:
                break
            c["div_score"] = 0
            selected.append(c)

    return selected[:n_target]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_demo(seed=42, ensemble_name="ensemble_v2_improved"):
    """
    Create 20 demo HDF5 files from the held-out test split.

    Composition:
      7 clear positives  (prob ≥ 0.65, true label = 1)
      7 clear negatives  (prob ≤ 0.35, true label = 0)
      3 borderline       (prob within ±0.10 of threshold, either label)
      2 edge cases       (extreme age OR low model agreement OR cross-label surprise)
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CHAGASVISION — DEMO SAMPLE CREATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Load metadata ──────────────────────────────────────────────
    meta_path = PROCESSED_DIR / "metadata.pkl"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found. Run preprocess_data.py first.")
        return

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    test_samples = metadata.get("test", [])
    if not test_samples:
        print("  ERROR: No test split in metadata.")
        return

    n_pos_test = sum(1 for s in test_samples if s["label"] == 1)
    n_neg_test = len(test_samples) - n_pos_test
    print(f"\n  Test split: {len(test_samples):,} samples")
    print(f"    Positive (Chagas): {n_pos_test:,}")
    print(f"    Negative (Normal): {n_neg_test:,}")
    print(f"  Source: {PROCESSED_DIR / 'metadata.pkl'}")
    print(f"  These samples were NEVER seen during training.")

    # ── Load ensemble ──────────────────────────────────────────────
    models, threshold = _load_ensemble(ensemble_name)
    if models is None:
        print("  ERROR: Could not load ensemble. Check models directory.")
        return

    # ── Load signal cache (memory-mapped) ─────────────────────────
    print(f"\n  Loading signal files (memory-mapped)...")
    cache = {}
    needed_files = set(s["file"] for s in test_samples)
    for fname in needed_files:
        fp = PROCESSED_DIR / fname
        if fp.exists():
            cache[fname] = np.load(fp, mmap_mode="r")
            print(f"    Loaded: {fname}  ({cache[fname].shape[0]:,} samples)")
        else:
            print(f"    MISSING: {fname}")

    # ── Run predictions on ALL test samples ────────────────────────
    print(f"\n  Running ensemble predictions on {len(test_samples):,} test samples...")
    print(f"  (This may take a few minutes on CPU)")

    random.seed(seed)
    np.random.seed(seed)

    # Pre-shuffle to avoid always picking first-indexed samples
    shuffled = list(test_samples)
    random.shuffle(shuffled)

    # Categorise all samples
    cat_buckets = {
        "clear_positive": [],
        "clear_negative": [],
        "borderline":     [],
        "edge":           [],
    }

    print(f"  Progress: ", end="", flush=True)
    for idx, s in enumerate(shuffled):
        if idx % 500 == 0:
            print(".", end="", flush=True)

        data = cache.get(s["file"])
        if data is None:
            continue

        signal = np.array(data[s["index"]], dtype=np.float32)
        prob, per_model, pred, agreement = _predict(models, signal, s["age"], s["sex"], threshold)

        if prob is None:
            continue

        entry = {
            **s,
            "prob":       round(prob, 4),
            "per_model":  [round(p, 4) for p in per_model],
            "pred":       pred,
            "agreement":  agreement,
            "correct":    (pred == "Chagas Positive") == (s["label"] == 1),
            "signal":     signal,
            "strength":   0.0,
        }

        # Categorise
        if s["label"] == 1 and prob >= CLEAR_POS_MIN:
            entry["strength"] = prob
            cat_buckets["clear_positive"].append(entry)

        elif s["label"] == 0 and prob <= CLEAR_NEG_MAX:
            entry["strength"] = 1 - prob
            cat_buckets["clear_negative"].append(entry)

        elif BORDER_LOW <= prob <= BORDER_HIGH:
            entry["strength"] = 1 - abs(prob - threshold)
            cat_buckets["borderline"].append(entry)

        # Edge cases: extreme age OR low agreement (split models)
        if (s["age"] <= EDGE_AGE_MAX or s["age"] >= EDGE_AGE_MIN) or agreement < 0.8:
            entry["edge_reason"] = []
            if s["age"] <= EDGE_AGE_MAX:
                entry["edge_reason"].append(f"young patient (age {s['age']})")
            if s["age"] >= EDGE_AGE_MIN:
                entry["edge_reason"].append(f"older patient (age {s['age']})")
            if agreement < 0.8:
                entry["edge_reason"].append(f"low model agreement ({agreement:.0%})")
            entry["edge_reason"] = "; ".join(entry["edge_reason"])
            entry["strength"] = 1 - agreement  # prefer most uncertain
            cat_buckets["edge"].append(entry)

    print(" done.")

    # Print category counts
    print(f"\n  Candidates found:")
    for cat, items in cat_buckets.items():
        print(f"    {cat:<20}: {len(items):,}")

    # ── Select with diversity ──────────────────────────────────────
    seen_sources, seen_ages, seen_sexes = set(), set(), set()
    final_selected = []

    targets = [
        ("clear_positive", 7,  False),
        ("clear_negative", 7,  False),
        ("borderline",     3,  True),   # prefer split agreement for borderline
        ("edge",           2,  True),
    ]

    for cat, n, prefer_split in targets:
        cands = cat_buckets[cat]
        if not cands:
            print(f"  WARNING: No candidates for category '{cat}'")
            continue

        chosen = select_diverse_samples(
            cands, n,
            seen_sources, seen_ages, seen_sexes,
            prefer_agreement_spread=prefer_split,
        )

        for c in chosen:
            c["category"] = cat
            final_selected.append(c)
            seen_sources.add(c["source"])
            seen_ages.add(_age_band(c["age"]))
            seen_sexes.add("M" if c["sex"] == 1 else "F")

    # Shuffle final order so positives aren't all together
    random.shuffle(final_selected)

    # ── Save HDF5 files + answer sheet ────────────────────────────
    print(f"\n  Saving {len(final_selected)} demo files to: {DEMO_DIR}")
    print(f"\n  {'ID':<15} {'Cat':<18} {'Prob':>6} {'True':>5} {'Agree':>6} {'Age':>4} {'Sex':>4} {'Source'}")
    print(f"  {'-'*15} {'-'*18} {'-'*6} {'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*12}")

    answer_sheet = []

    for rank, entry in enumerate(final_selected):
        pid   = f"patient_{rank + 1:03d}"
        fname = f"{pid}.h5"

        # Save HDF5
        out_path = DEMO_DIR / fname
        with h5py.File(out_path, "w") as f:
            f.create_dataset("tracings", data=entry["signal"].astype(np.float32))
            f.attrs["patient_id"] = pid
            f.attrs["age"]        = int(entry["age"])
            f.attrs["sex"]        = "Male" if entry["sex"] == 1 else "Female"
            # No label stored — keeps demo blind

        true_label = "CHAGAS POSITIVE" if entry["label"] == 1 else "CHAGAS NEGATIVE"
        sex_str    = "M" if entry["sex"] == 1 else "F"
        cat_short  = entry["category"].replace("_", " ").upper()

        print(f"  {pid:<15} {cat_short:<18} {entry['prob']:>6.3f} "
              f"{'POS' if entry['label']==1 else 'NEG':>5} "
              f"{entry['agreement']:>6.0%} {entry['age']:>4} {sex_str:>4}  {entry['source']}")

        record = {
            "patient_id":    pid,
            "filename":      fname,
            "category":      entry["category"],
            "true_label":    true_label,
            "label_code":    entry["label"],
            "model_prob":    entry["prob"],
            "model_pred":    entry["pred"],
            "per_model_probs": entry["per_model"],
            "agreement":     entry["agreement"],
            "correct":       entry["correct"],
            "age":           int(entry["age"]),
            "sex":           "Male" if entry["sex"] == 1 else "Female",
            "source":        entry["source"],
            "age_band":      _age_band(entry["age"]),
            "edge_reason":   entry.get("edge_reason", ""),
        }
        answer_sheet.append(record)

    # ── Save answer sheet (secret) ─────────────────────────────────
    answer_data = {
        "WARNING":        "DO NOT SHOW DURING VIVA — EXAMINER ANSWER KEY",
        "created":        datetime.now().isoformat(),
        "ensemble":       ensemble_name,
        "threshold":      threshold,
        "seed":           seed,
        "total_samples":  len(answer_sheet),
        "composition": {
            "clear_positive": sum(1 for r in answer_sheet if r["category"] == "clear_positive"),
            "clear_negative": sum(1 for r in answer_sheet if r["category"] == "clear_negative"),
            "borderline":     sum(1 for r in answer_sheet if r["category"] == "borderline"),
            "edge":           sum(1 for r in answer_sheet if r["category"] == "edge"),
        },
        "accuracy_on_demo": round(sum(1 for r in answer_sheet if r["correct"]) / len(answer_sheet), 4),
        "samples": answer_sheet,
    }

    secret_path = DEMO_DIR / "ANSWER_SHEET_SECRET.json"
    with open(secret_path, "w") as f:
        json.dump(answer_data, f, indent=2)

    # ── Save printable answer key ──────────────────────────────────
    key_lines = [
        "CHAGASVISION - VIVA ANSWER KEY (SECRET)",
        "=" * 60,
        f"Created    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Ensemble   : {ensemble_name}",
        f"Threshold  : {threshold:.3f}",
        f"Data source: Held-out test split only (never seen in training)",
        f"Accuracy   : {answer_data['accuracy_on_demo']:.1%} on these 20 samples",
        "",
        f"{'ID':<15} {'Age':<5} {'Sex':<5} {'Category':<18} {'TRUE LABEL':<20} {'Prob':>6} {'Correct':>8}",
        "-" * 80,
    ]
    for r in answer_sheet:
        correct_str = "CORRECT" if r["correct"] else "WRONG"
        key_lines.append(
            f"{r['patient_id']:<15} {r['age']:<5} {r['sex']:<5} "
            f"{r['category']:<18} {r['true_label']:<20} "
            f"{r['model_prob']:>6.3f} {correct_str:>8}"
        )

    key_lines += [
        "",
        "EDGE CASE NOTES:",
    ]
    for r in answer_sheet:
        if r.get("edge_reason"):
            key_lines.append(f"  {r['patient_id']}: {r['edge_reason']}")

    with open(DEMO_DIR / "ANSWER_KEY.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(key_lines))

    # ── Summary ────────────────────────────────────────────────────
    comp = answer_data["composition"]
    print(f"\n{'='*65}")
    print(f"  DEMO CREATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Files saved to : {DEMO_DIR}")
    print(f"  Total samples  : {len(answer_sheet)}")
    print(f"    Clear positive : {comp['clear_positive']}")
    print(f"    Clear negative : {comp['clear_negative']}")
    print(f"    Borderline     : {comp['borderline']}")
    print(f"    Edge cases     : {comp['edge']}")
    print(f"  Model accuracy on demo set: {answer_data['accuracy_on_demo']:.1%}")
    print(f"\n  SECRET files (do not share during viva):")
    print(f"    {secret_path.name}")
    print(f"    ANSWER_KEY.txt")
    print(f"\n  IMPORTANT: All samples drawn from held-out test split.")
    print(f"  The model has NEVER seen these during training.")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY EXISTING FILES
# ══════════════════════════════════════════════════════════════════════════════

def verify_demo(ensemble_name="ensemble_v2_improved"):
    """Re-run predictions on existing demo HDF5 files and compare to answer sheet."""
    models, threshold = _load_ensemble(ensemble_name)
    if models is None:
        return

    h5_files = sorted(DEMO_DIR.glob("patient_*.h5"))
    if not h5_files:
        print(f"  No demo files found in {DEMO_DIR}")
        return

    secret_path = DEMO_DIR / "ANSWER_SHEET_SECRET.json"
    answer_sheet = {}
    if secret_path.exists():
        with open(secret_path) as f:
            data = json.load(f)
        answer_sheet = {r["filename"]: r for r in data["samples"]}

    print(f"\n  {'File':<15} {'Prob':>6} {'Pred':<18} {'True':>16} {'Match':>6}")
    print(f"  {'-'*15} {'-'*6} {'-'*18} {'-'*16} {'-'*6}")

    correct = 0
    for h5f in h5_files:
        with h5py.File(h5f, "r") as f:
            sig = np.array(f["tracings"], dtype=np.float32)
            age = int(f.attrs.get("age", 50))
            sex = 1 if f.attrs.get("sex", "Female") == "Male" else 0

        prob, _, pred, _ = _predict(models, sig, age, sex, threshold)
        ans = answer_sheet.get(h5f.name, {})
        true_label = ans.get("true_label", "UNKNOWN")
        match = "✓" if ans and (pred == ans.get("model_pred")) else "?"
        if match == "✓":
            correct += 1

        print(f"  {h5f.stem:<15} {prob:>6.3f} {pred:<18} {true_label:>16} {match:>6}")

    if answer_sheet:
        print(f"\n  Consistency with saved predictions: {correct}/{len(h5_files)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChagasVision Demo Sample Creator")
    parser.add_argument("--verify",   action="store_true", help="Verify existing demo files")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed for selection")
    parser.add_argument("--ensemble", default="ensemble_v2_improved", help="Ensemble name")
    args = parser.parse_args()

    if args.verify:
        verify_demo(args.ensemble)
    else:
        create_demo(seed=args.seed, ensemble_name=args.ensemble)