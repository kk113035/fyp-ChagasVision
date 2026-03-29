#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 

import gc
import pickle
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from config import (
    SAMI_HDF5, SAMI_CSV,
    CODE15_HDF5_FILES, CODE15_LABELS, CODE15_DEMO_CSV,
    PTBXL_DIR, PTBXL_CSV,
    PROCESSED_DIR, SEQUENCE_LENGTH, SAMPLING_RATE, NUM_LEADS,
)
from preprocessing import ECGPreprocessor


def process_samitrop(pp: ECGPreprocessor, max_n=None):
    """All 1 631 samples are Chagas positive."""
    print("\n" + "=" * 60 + "\n  SAMI-TROP\n" + "=" * 60)
    if not SAMI_HDF5.exists():
        print(f"  SKIP — file not found: {SAMI_HDF5}")
        return []

    demo = pd.read_csv(SAMI_CSV)
    samples, signals = [], []

    with h5py.File(SAMI_HDF5, "r") as f:
        n = f["tracings"].shape[0]
        if max_n: n = min(n, max_n)
        for i in tqdm(range(n), desc="  SaMi-Trop"):
            raw = np.array(f["tracings"][i], dtype=np.float32)
            proc = pp.process(raw)
            age = int(demo.iloc[i].get("age", 50)) if i < len(demo) else 50
            sex = 1 if (i < len(demo) and demo.iloc[i].get("is_male", False)) else 0
            signals.append(proc)
            samples.append({"index": len(signals) - 1, "label": 1,
                            "age": age, "sex": sex, "source": "samitrop",
                            "file": "samitrop_processed.npy"})

    out = PROCESSED_DIR / "samitrop_processed.npy"
    np.save(out, np.array(signals, dtype=np.float32))
    print(f"  Saved {len(signals)} samples → {out.name}")
    return samples


def process_code15(pp: ECGPreprocessor, max_n_per_file=None):
    """CODE-15 — using 6 HDF5 files (expandable to all 15)."""
    print("\n" + "=" * 60 + "\n  CODE-15\n" + "=" * 60)
    if not CODE15_LABELS.exists():
        print(f"  SKIP — labels not found: {CODE15_LABELS}")
        return []

    labels_df = pd.read_csv(CODE15_LABELS)
    exam2chagas = dict(zip(labels_df["exam_id"].astype(int), labels_df["chagas"].astype(bool)))
    demo_df = pd.read_csv(CODE15_DEMO_CSV) if CODE15_DEMO_CSV.exists() else pd.DataFrame()
    exam2age = dict(zip(demo_df["exam_id"].astype(int), demo_df["age"])) if len(demo_df) else {}
    exam2sex = dict(zip(demo_df["exam_id"].astype(int), demo_df["is_male"])) if len(demo_df) else {}

    print(f"  Chagas labels CSV: {len(exam2chagas):,} exam_ids")
    print(f"  Demographics CSV:  {len(exam2age):,} exam_ids")
    print(f"  (Both CSVs cover ALL 15 parts — your 6 files are a subset)")

    all_samples = []

    for fi, hdf5_path in enumerate(CODE15_HDF5_FILES):
        if not hdf5_path.exists():
            print(f"  SKIP — {hdf5_path.name} not found")
            continue

        signals, samples = [], []
        n_meta_matched = 0
        n_meta_defaulted = 0
        with h5py.File(hdf5_path, "r") as f:
            eids = np.array(f["exam_id"])
            n = len(eids)
            if max_n_per_file: n = min(n, max_n_per_file)
            for i in tqdm(range(n), desc=f"  part{fi}"):
                eid = int(eids[i])
                if eid not in exam2chagas:
                    continue

                raw = np.array(f["tracings"][i], dtype=np.float32)
                proc = pp.process(raw)

                # ── Metadata lookup ────────────────────────────────
                # exams.csv and code15_chagas_labels.csv cover ALL 15
                # parts of CODE-15, not just the HDF5 files we load.
                # So exam_ids from any of our 6 files will have matching
                # rows in the CSVs.  Default values are only needed for
                # rare data-quality edge cases (corrupt/missing exam_id).
                age = int(exam2age.get(eid, 50))
                sex = 1 if exam2sex.get(eid, False) else 0

                if eid in exam2age:
                    n_meta_matched += 1
                else:
                    n_meta_defaulted += 1
                label = 1 if exam2chagas[eid] else 0
                signals.append(proc)
                samples.append({"index": len(signals) - 1, "label": label,
                                "age": age, "sex": sex, "source": "code15",
                                "file": f"code15_part{fi}_processed.npy"})

        out = PROCESSED_DIR / f"code15_part{fi}_processed.npy"
        np.save(out, np.array(signals, dtype=np.float32))
        n_pos = sum(1 for s in samples if s["label"] == 1)
        print(f"  Saved {len(signals)} samples ({n_pos} pos) → {out.name}")
        print(f"    Metadata: {n_meta_matched} matched, {n_meta_defaulted} defaulted"
              f" ({100*n_meta_matched/max(n_meta_matched+n_meta_defaulted,1):.1f}% hit rate)")
        all_samples.extend(samples)
        del signals; gc.collect()

    return all_samples


def process_ptbxl(pp: ECGPreprocessor, max_n=None):
    """PTB-XL — all assumed Chagas negative (German database)."""
    print("\n" + "=" * 60 + "\n  PTB-XL\n" + "=" * 60)
    if not PTBXL_CSV.exists():
        print(f"  SKIP — CSV not found: {PTBXL_CSV}")
        return []

    try:
        import wfdb
    except ImportError:
        print("  SKIP — wfdb not installed (pip install wfdb)")
        return []

    df = pd.read_csv(PTBXL_CSV)
    if max_n: df = df.head(max_n)

    signals, samples, errors = [], [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  PTB-XL"):
        eid = int(row["ecg_id"])
        subfolder = f"{(eid // 1000) * 1000:05d}"
        rec_path = PTBXL_DIR / subfolder / f"{eid:05d}_hr"
        if not Path(str(rec_path) + ".hea").exists():
            errors += 1; continue
        try:
            record = wfdb.rdrecord(str(rec_path))
            raw = record.p_signal.T  # [12, N]
            proc = pp.process(raw)
            age = int(row["age"]) if pd.notna(row.get("age")) else 50
            sex = 0 if row.get("sex", 0) == 1 else 1  # PTB-XL convention flipped
            signals.append(proc)
            samples.append({"index": len(signals) - 1, "label": 0,
                            "age": age, "sex": sex, "source": "ptbxl",
                            "file": "ptbxl_processed.npy"})
        except Exception:
            errors += 1

    if signals:
        out = PROCESSED_DIR / "ptbxl_processed.npy"
        np.save(out, np.array(signals, dtype=np.float32))
        print(f"  Saved {len(signals)} samples → {out.name}  (errors: {errors})")
    return samples


def create_splits(samples, test_frac=0.15, seed=42):
    """
    Stratified dev/test split.

    Why only TWO splits (not three)?
    ─────────────────────────────────
    K-fold CV creates its own train/val splits WITHIN the dev set.
    A pre-split 'val' would be redundant — K-fold already validates
    every sample exactly once across the 5 folds.

    The test set is sacred: never seen during training, threshold
    optimisation, or model selection.  It's used ONLY for final
    evaluation (validation.py) and examiner demo files (demo.py).

    Split:
        dev  = 85% of data → used for 5-fold cross-validation
        test = 15% of data → held out for final honest evaluation

    Both splits are stratified to preserve the ~3.4% positive rate.

    Reference: Varma & Simon (2006) 'Bias in error estimation when
    using cross-validation for model selection', BMC Bioinformatics.
    """
    rng = np.random.RandomState(seed)
    pos = [s for s in samples if s["label"] == 1]
    neg = [s for s in samples if s["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    n_test_p = max(1, int(len(pos) * test_frac))
    n_test_n = int(len(neg) * test_frac)

    test = pos[:n_test_p] + neg[:n_test_n]
    dev  = pos[n_test_p:] + neg[n_test_n:]

    rng.shuffle(test)
    rng.shuffle(dev)
    return dev, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ptbxl", action="store_true")
    parser.add_argument("--max", type=int, default=None, help="Max samples per dataset")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pp = ECGPreprocessor()

    print("=" * 60)
    print("  CHAGASVISION DATA PREPROCESSING")
    print("=" * 60)

    all_samples = []
    all_samples.extend(process_samitrop(pp, args.max))
    all_samples.extend(process_code15(pp, args.max))
    if not args.skip_ptbxl:
        all_samples.extend(process_ptbxl(pp, args.max))

    n_pos = sum(1 for s in all_samples if s["label"] == 1)
    print(f"\n  TOTAL: {len(all_samples):,}  ({n_pos:,} positive, "
          f"{len(all_samples) - n_pos:,} negative)")

    dev, test = create_splits(all_samples)

    n_dev_pos  = sum(1 for s in dev if s["label"] == 1)
    n_test_pos = sum(1 for s in test if s["label"] == 1)

    print(f"\n  Splits (stratified):")
    print(f"    Dev set  : {len(dev):,}  ({n_dev_pos:,} positive = {100*n_dev_pos/len(dev):.1f}%)  → 5-fold CV")
    print(f"    Test set : {len(test):,}  ({n_test_pos:,} positive = {100*n_test_pos/len(test):.1f}%)  → final evaluation ONLY")

    metadata = {
        "all_samples": all_samples,
        "train": dev,           # training.py reads this as dev_samples
        "test": test,           # validation.py and demo.py read this
        "config": {"sequence_length": SEQUENCE_LENGTH, "sampling_rate": SAMPLING_RATE},
    }
    with open(PROCESSED_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n  Saved metadata → {PROCESSED_DIR / 'metadata.pkl'}")


if __name__ == "__main__":
    main()
