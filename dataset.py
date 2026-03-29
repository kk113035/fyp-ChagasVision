#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 



import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional


def add_gaussian_noise(signal: np.ndarray, std: float = 0.03) -> np.ndarray:
    
    noise = np.random.normal(0, std, signal.shape).astype(np.float32)
    return signal + noise


def amplitude_scale(signal: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    
    scale_factor = np.random.uniform(lo, hi)
    return signal * scale_factor


def time_shift(signal: np.ndarray, max_shift: int = 50) -> np.ndarray:
    
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(signal, shift, axis=1)


def lead_dropout(signal: np.ndarray, max_leads: int = 2) -> np.ndarray:
    
    signal = signal.copy()
    n_drop = np.random.randint(1, max_leads + 1)
    leads_to_drop = np.random.choice(12, size=n_drop, replace=False)
    signal[leads_to_drop] = 0.0
    return signal


def baseline_wander(signal: np.ndarray, fs: int = 400) -> np.ndarray:
    
    t = np.arange(signal.shape[1], dtype=np.float32) / fs
    freq = np.random.uniform(0.1, 0.5)
    amp = np.random.uniform(0.02, 0.1)
    phase = np.random.uniform(0, 2 * np.pi)
    wander = amp * np.sin(2 * np.pi * freq * t + phase)
    return signal + wander


def time_mask(signal: np.ndarray, max_width: int = 100) -> np.ndarray:
    
    signal = signal.copy()
    L = signal.shape[1]
    w = np.random.randint(1, min(max_width + 1, L))
    start = np.random.randint(0, L - w)
    signal[:, start:start + w] = 0.0
    return signal


def augment_ecg(signal: np.ndarray, label: int) -> np.ndarray:
    
   
    if label == 0 and np.random.random() > 0.2:
        return signal

    if label == 1 and np.random.random() > 0.8:
        return signal

    s = signal.copy()

    if np.random.random() < 0.5:    
        s = add_gaussian_noise(s)     

    if np.random.random() < 0.5:    
        s = amplitude_scale(s)        

    if np.random.random() < 0.3:    
        s = time_shift(s)            

    if np.random.random() < 0.15:   
        s = lead_dropout(s)           

    if np.random.random() < 0.3:    
        s = baseline_wander(s)        

    if np.random.random() < 0.3:    
        s = time_mask(s)              

    return np.clip(s, -5.0, 5.0).astype(np.float32)


class ChagasDataset(Dataset):
    

    def __init__(
        self,
        samples: List[Dict],
        processed_dir: Path,
        training: bool = True,
        augment: bool = True,
        data_cache: Optional[Dict] = None,
    ):
       
        self.samples = samples
        self.processed_dir = Path(processed_dir)
        self.training = training
        self.augment = augment and training  # only augment during training
        self.cache = data_cache if data_cache is not None else {}
        self._preload()

    def _preload(self):
        """Memory-map all needed .npy files for fast random access."""
        files_needed = set(s["file"] for s in self.samples)
        for fname in files_needed:
            if fname not in self.cache:
                fp = self.processed_dir / fname
                if fp.exists():
                    
                    self.cache[fname] = np.load(fp, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            data = self.cache.get(s["file"])
            if data is not None:
                signal = np.array(data[s["index"]], dtype=np.float32)
            else:
                signal = np.zeros((12, 2048), dtype=np.float32)

            if self.augment:
                signal = augment_ecg(signal, s["label"])

            return (
                torch.from_numpy(signal),
                torch.tensor(s["label"], dtype=torch.float32),
                torch.tensor(s["age"],   dtype=torch.float32),
                torch.tensor(s["sex"],   dtype=torch.long),
            )

        except Exception:
            return (
                torch.zeros(12, 2048),
                torch.tensor(0.0),
                torch.tensor(50.0),
                torch.tensor(0),
            )
