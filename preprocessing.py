"""
ECG Preprocessing Pipeline
===========================
Standardises raw ECG signals for model input.

Steps (in order)
-----------------
1. Amplitude clipping ±5 mV                    (artifact removal)
2. Resampling to 2048 samples                  (fixed-length input)
3. Bandpass filter 0.5-45 Hz, 4th-order Butterworth  (baseline wander + HF noise)
4. Notch filter 60 Hz, Q=30                    (powerline interference)
5. Z-score normalisation per lead              (zero-centred, unit variance)
6. Clip to ±5 std                              (outlier suppression)

References
----------
[1] Kligfield et al. (2007) AHA/ACC/HRS ECG standardisation, Circulation.
[2] Luo & Johnston (2010) ECG filtering review, J Electrocardiology.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
from config import SAMPLING_RATE, SEQUENCE_LENGTH, NUM_LEADS


class ECGPreprocessor:
    """Stateless ECG preprocessing — filter coefficients computed once."""

    def __init__(self, fs: int = SAMPLING_RATE, target_len: int = SEQUENCE_LENGTH):
        self.fs = fs
        self.target_len = target_len

        nyq = fs / 2.0
        self.bp_b, self.bp_a = butter(4, [0.5 / nyq, 45.0 / nyq], btype="band")
        self.notch_b, self.notch_a = iirnotch(60.0, Q=30, fs=fs)

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Full pipeline: raw → model-ready tensor.

        Args:
            signal: [12, N] or [N, 12] raw ECG
        Returns:
            [12, 2048] preprocessed float32 array
        """
        # Ensure [leads, samples]
        if signal.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {signal.shape}")
        if signal.shape[0] != NUM_LEADS:
            signal = signal.T
        if signal.shape[0] != NUM_LEADS:
            raise ValueError(f"Cannot reshape to {NUM_LEADS} leads: {signal.shape}")

        signal = np.asarray(signal, dtype=np.float32)

        # 1. Amplitude clipping
        signal = np.clip(signal, -5.0, 5.0)

        # 2. Resample to target length
        if signal.shape[1] != self.target_len:
            resampled = np.zeros((NUM_LEADS, self.target_len), dtype=np.float32)
            for i in range(NUM_LEADS):
                resampled[i] = resample(signal[i], self.target_len)
            signal = resampled

        # 3-4. Bandpass + notch filtering
        for i in range(NUM_LEADS):
            try:
                signal[i] = filtfilt(self.bp_b, self.bp_a, signal[i])
                signal[i] = filtfilt(self.notch_b, self.notch_a, signal[i])
            except ValueError:
                pass  # keep unfiltered if signal too short for filter

        # 5. Z-score normalisation per lead
        for i in range(NUM_LEADS):
            mu, sigma = signal[i].mean(), signal[i].std()
            if sigma > 1e-8:
                signal[i] = (signal[i] - mu) / sigma

        # 6. Clip outliers
        signal = np.clip(signal, -5.0, 5.0)
        return signal.astype(np.float32)
