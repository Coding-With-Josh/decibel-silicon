"""Mel-spaced band grouping of the 65-bin rFFT magnitude grid (N=128 @ 16 kHz).

Why bands at all: RNNoise's transferable insight is that a small model can
output *band-level* gains (perceptually meaningful resolution) instead of
per-bin gains, which keeps the GRU tiny while still destroying noise between
spectral lines. This module fixes ONE deterministic bin->band assignment so
training, the numpy inference path, and the power manifest all agree.

Band edges are evenly spaced in mel from 0 to mel(8000 Hz): 24 bands. Each
bin k (center freq k*125 Hz) is assigned to the band whose mel range contains
it. Assignment is non-overlapping (each bin in exactly one band), unlike a
full mel filterbank with overlap - the extra correlation is unnecessary for a
gain mask and would cost MACs on the eventual fixed-point port.

The mapping is exported with the weights (npz `band_map`), so a change here
invalidates old checkpoints rather than silently reinterpreting them.
"""

from __future__ import annotations

import numpy as np

from .constants import FS, N_BANDS, N_FFT


def _hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_edges_hz(n_bands: int = N_BANDS, fs: int = FS) -> np.ndarray:
    """n_bands+1 mel-spaced edge frequencies from 0 to Nyquist."""
    nyq = fs / 2.0
    mel_lo, mel_hi = _hz_to_mel(0.0), _hz_to_mel(nyq)
    mel_edges = np.linspace(mel_lo, mel_hi, n_bands + 1)
    return np.array([_mel_to_hz(m) for m in mel_edges], dtype=np.float64)


def build_bin_to_band(n_fft: int = N_FFT, fs: int = FS,
                      n_bands: int = N_BANDS) -> np.ndarray:
    """Map each of n_fft//2+1 rFFT bins to a band index in 0..n_bands-1.

    Mel edges are converted to bin counts and then coerced to be strictly
    increasing with at least ONE bin per band: raw mel spacing at 16 kHz/24
    bands leaves a zero-count band in the linear low-frequency region
    (observed: band 3 empty), which would make band_rms/expand_band_gains
    throw. The coercion is deterministic (maximum.accumulate + propagate
    +1), so the exported band_map stays a stable canonical artifact.
    """
    edges = mel_edges_hz(n_bands, fs)
    n_bins = n_fft // 2 + 1
    edges_bin = np.clip(
        np.round(edges / (fs / n_fft)).astype(np.int64), 0, n_bins)
    edges_bin = np.maximum.accumulate(edges_bin)   # monotone
    edges_bin[0] = 0
    for i in range(1, n_bands):
        if edges_bin[i] <= edges_bin[i - 1]:
            edges_bin[i] = edges_bin[i - 1] + 1    # >= 1 bin per band
    edges_bin = np.clip(edges_bin, 0, n_bins)
    edges_bin[-1] = n_bins                          # exclusive end at Nyquist

    bin_to_band = np.empty(n_bins, dtype=np.int64)
    for k in range(n_bins):
        band = int(np.searchsorted(edges_bin, k + 1, side="left") - 1)
        bin_to_band[k] = int(np.clip(band, 0, n_bands - 1))
    counts = np.bincount(bin_to_band, minlength=n_bands)
    if (counts == 0).any():
        raise RuntimeError(f"some mel bands are empty: {counts.tolist()}")
    return bin_to_band


def band_rms(x_bins: np.ndarray, bin_to_band: np.ndarray,
             n_bands: int) -> np.ndarray:
    """Per-band RMS magnitude: sqrt(mean_k |x_k|^2) over each band's bins.

    x_bins: (..., n_bins) magnitude spectrum. Returns (..., n_bands).
    Band widths vary; the ratio form used by the IRM target is
    scale-invariant per band (clean and noise share the same banding).
    """
    arr = np.asarray(x_bins, dtype=np.float64)
    out = np.empty(arr.shape[:-1] + (n_bands,), dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    for b in range(n_bands):
        idx = np.where(bin_to_band == b)[0]
        if idx.size == 0:
            raise RuntimeError(f"band {b} has no bins")
        out[..., b] = np.sqrt(np.mean(flat[:, idx] ** 2, axis=1)).reshape(
            arr.shape[:-1])
    return out


def expand_band_gains(band_gains: np.ndarray, bin_to_band: np.ndarray,
                      n_bands: int) -> np.ndarray:
    """Repeat each band gain across its bins -> per-bin gain array."""
    band_gains = np.asarray(band_gains, dtype=np.float64)
    out = np.empty(band_gains.shape[:-1] + (bin_to_band.size,),
                   dtype=np.float64)
    for b in range(n_bands):
        idx = np.where(bin_to_band == b)[0]
        out[..., idx] = band_gains[..., b:b + 1]
    return out


# Canonical mapping (module-level cache; deterministic).
BIN_TO_BAND = build_bin_to_band()