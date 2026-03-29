#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


# PROJECT_ROOT = Path(r"D:\Kaveesha - IIT\4th Year\FYP\ChagasVisionProto")
PROJECT_ROOT = Path(__file__).resolve().parent

# Raw datasets
SAMI_HDF5           = PROJECT_ROOT / "dataRaw" / "sami trop" / "exams" / "exams.hdf5"
SAMI_CSV            = PROJECT_ROOT / "dataRaw" / "sami trop" / "exams.csv"
CODE15_DIR          = PROJECT_ROOT / "dataRaw" / "code15"
CODE15_LABELS       = CODE15_DIR / "code15_chagas_labels.csv"
CODE15_DEMO_CSV     = CODE15_DIR / "exams.csv"    # demographics
CODE15_HDF5_FILES   = [CODE15_DIR / f"exams_part{i}" / f"exams_part{i}.hdf5" for i in range(6)]
PTBXL_DIR           = PROJECT_ROOT / "dataRaw" / "ptb_xl_400hz"
PTBXL_CSV           = PTBXL_DIR / "ptbxl_database.csv"

# Output directories
PROCESSED_DIR = PROJECT_ROOT / "dataProcessed"
MODELS_DIR    = PROJECT_ROOT / "models_improved"
RESULTS_DIR   = PROJECT_ROOT / "results"

#ECG 

SAMPLING_RATE   = 400           # Hz (standard for SaMi-Trop / CODE-15)
SEQUENCE_LENGTH = 2048          # samples → 5.12 s at 400 Hz
NUM_LEADS       = 12
LEAD_NAMES      = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']




@dataclass
class ModelConfig:
    
    num_leads: int              = NUM_LEADS
    seq_length: int             = SEQUENCE_LENGTH
    ms_out_channels: int        = 32            
    kernel_sizes: Tuple[int,...] = (3, 7, 15)   
    conv2_out: int              = 128           
    conv2_kernel: int           = 7             
    use_se: bool                = True          
    se_reduction: int           = 16           
    d_model: int                = 128           
    nhead: int                  = 4             
    num_transformer_layers: int = 2             
    dim_feedforward: int        = 512           
    use_metadata: bool          = True          
    metadata_dim: int           = 8             
    dropout: float              = 0.4           


@dataclass
class TrainingConfig:
    
    experiment_name: str   = "ensemble_v2_improved"
    n_folds: int           = 5
    max_epochs: int        = 50              
    batch_size: int        = 16              
    learning_rate: float   = 5e-4            
    weight_decay: float    = 1e-4            
    patience: int          = 7               
    gradient_clip: float   = 1.0             
    oversample_factor: int = 20              
    focal_gamma: float     = 2.0             
    label_smoothing: float = 0.1             
    seed: int              = 42
    use_augmentation: bool = True


@dataclass
class AugmentConfig:
    
    noise_prob: float       = 0.5
    noise_std_range: Tuple[float, float] = (0.01, 0.05)
    scale_prob: float       = 0.5
    scale_range: Tuple[float, float]     = (0.8, 1.2)
    shift_prob: float       = 0.3
    shift_max: int          = 100
    lead_drop_prob: float   = 0.2
    lead_drop_max: int      = 2
    baseline_wander_prob: float = 0.3
    time_mask_prob: float   = 0.3
    time_mask_width: int    = 100



CHAGAS_PATTERNS = {
    "rbbb": {
        "name": "Right Bundle Branch Block",
        "leads": ["V1", "V2", "V6"],
        "description": "Wide QRS with RSR' in V1/V2, slurred S in V6",
        "relevance": "Most common in Chagas cardiomyopathy (40-50%)",
        "reference": "Rojas et al. PLoS NTD 2018 — OR 4.60",
    },
    "lafb": {
        "name": "Left Anterior Fascicular Block",
        "leads": ["I", "aVL", "II", "III"],
        "description": "Left axis deviation with qR in aVL",
        "relevance": "Frequent in Chagas disease (30-40%)",
        "reference": "Rojas et al. PLoS NTD 2018 — OR 1.60",
    },
    "av_block": {
        "name": "Atrioventricular Block",
        "leads": ["II", "V1"],
        "description": "Prolonged PR interval or dropped beats",
        "relevance": "Indicates conduction damage (15-25%)",
        "reference": "Nunes et al. Circulation 2018",
    },
    "lvh": {
        "name": "Left Ventricular Hypertrophy",
        "leads": ["V5", "V6", "I", "aVL"],
        "description": "Tall R waves in lateral leads",
        "relevance": "Secondary to Chagas cardiomyopathy",
        "reference": "Ribeiro et al. Nat Rev Cardiol 2012",
    },
    "low_voltage": {
        "name": "Low Voltage QRS",
        "leads": LEAD_NAMES,
        "description": "Reduced QRS amplitude across leads",
        "relevance": "Indicates myocardial fibrosis (10-20%)",
        "reference": "Nunes et al. Circulation 2018",
    },
}



LITERATURE_BASELINES = {
    "CinC_2025_baseline": {
        "score": 0.691,
        "metric": "balanced_accuracy",
        "reference": "Moody (2025) CinC Challenge example algorithm",
        "dataset": "CODE-15",
        "note": "Baseline example code from PhysioNet; scored with balanced accuracy",
    },
    # ── Compare using AUC-ROC ──
    "Jidling_2023_REDSII": {
        "score": 0.82,
        "metric": "auroc",
        "reference": "Jidling et al. (2023) PLoS NTD",
        "dataset": "REDS-II (external validation)",
        "note": "AUC-ROC on external REDS-II dataset",
    },
    "Jidling_2023_ELSA": {
        "score": 0.77,
        "metric": "auroc",
        "reference": "Jidling et al. (2023) PLoS NTD",
        "dataset": "ELSA-Brasil (external validation)",
        "note": "AUC-ROC on external ELSA-Brasil dataset",
    },
    "Ribeiro_2020": {
        "score": 0.870,
        "metric": "auroc",
        "reference": "Ribeiro et al. (2020) Nature Communications",
        "dataset": "CODE (full, not directly comparable)",
        "note": "Uses full CODE dataset with >2M records",
    },
}


def verify_paths() -> bool:
    """Check all required data files exist."""
    checks = [
        ("SaMi-Trop HDF5", SAMI_HDF5),
        ("SaMi-Trop CSV", SAMI_CSV),
        ("CODE-15 Labels", CODE15_LABELS),
        ("CODE-15 Demographics", CODE15_DEMO_CSV),
        ("PTB-XL CSV", PTBXL_CSV),
        ("PTB-XL Directory", PTBXL_DIR),
    ]
    ok = True
    for name, p in checks:
        exists = p.exists()
        ok &= exists
        print(f"  {'OK' if exists else 'MISSING':>7}  {name}")

    print("\n  CODE-15 HDF5 files:")
    for p in CODE15_HDF5_FILES:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING':>7}  {p.name}")

    # Check a sample PTB-XL record to confirm folder structure
    print("\n  PTB-XL records (spot check):")
    sample_record = PTBXL_DIR / "00000" / "00001_hr.hea"
    exists = sample_record.exists()
    print(f"  {'OK' if exists else 'MISSING':>7}  00000/00001_hr.hea (first record)")
    if not exists:
        print(f"         Expected at: {sample_record}")
        print(f"         Make sure PTB-XL 400Hz version is in: {PTBXL_DIR}")

    # Check preprocessed data (if already run)
    meta = PROCESSED_DIR / "metadata.pkl"
    print(f"\n  Preprocessed data:")
    print(f"  {'OK' if meta.exists() else 'NOT YET':>7}  metadata.pkl {'(ready for training)' if meta.exists() else '(run preprocess_data.py first)'}")

    return ok


if __name__ == "__main__":
    verify_paths()