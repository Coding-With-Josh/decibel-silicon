"""Numpy-only inference + the pipeline gain hook for the learned model.

The hook is the ONLY bridge between the learned weights and the classical
SpectralSubtraction chain: gain_hook(mag, noise_mag, frame_index) replaces
the alpha/Berouti gain decision per ACTIVE frame while every other stage
(window, FFT, noise estimator, OLA reconstruction) stays the classical code.

FAIL-CLOSED load: the npz must match the export schema exactly (shapes,
finiteness, band map, metadata). Any mismatch raises; nothing is clamped to
a plausible number to keep a cell alive (Phase 2 rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.spectral_subtraction import DEFAULT_FS, DEFAULT_HOP, DEFAULT_N_FFT
from .banding import BIN_TO_BAND, band_rms, expand_band_gains
from .constants import HIDDEN, N_BANDS
from .model import LearnedWeights, gru_forward_numpy

FEATURE_EPS = 1e-8


@dataclass
class LoadedGainModel:
    """Validated in-memory weights + metadata (loaded exactly once)."""

    weights: LearnedWeights
    meta: dict
    path: str

    def describe(self) -> str:
        m = self.meta
        return (f"learned GRU gain model: hidden={m['hidden']}, "
                f"n_features={m['n_features']}, n_bands={m['n_bands']}, "
                f"trained_on={m['trained_on']}, held_out={m['held_out']}")


def load_model_npz(path: str | Path) -> LoadedGainModel:
    """Load and schema-validate the float32 export. Raises on any mismatch."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--gain-model not found: {path}")
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - wrap any numpy load failure
        raise ValueError(f"--gain-model {path} is not a readable npz: {exc}")

    required = ("w_ir", "w_hr", "b_ih", "b_hh", "out_w", "out_b",
                "band_map", "meta")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(f"--gain-model {path} missing arrays {missing}")

    meta = json.loads(str(d["meta"]))
    for key in ("n_fft", "hop", "fs", "n_bands", "hidden", "n_features",
                "trained_on", "held_out"):
        if key not in meta:
            raise ValueError(f"--gain-model {path} meta missing {key}")
    if int(meta["n_fft"]) != DEFAULT_N_FFT or int(meta["hop"]) != DEFAULT_HOP \
            or int(meta["fs"]) != DEFAULT_FS:
        raise ValueError(
            f"--gain-model geometry mismatch: expects N={DEFAULT_N_FFT}, "
            f"hop={DEFAULT_HOP}, fs={DEFAULT_FS}; model says "
            f"{meta['n_fft']}/{meta['hop']}/{meta['fs']}")

    w_ir = np.asarray(d["w_ir"], dtype=np.float64)
    w_hr = np.asarray(d["w_hr"], dtype=np.float64)
    b_ih = np.asarray(d["b_ih"], dtype=np.float64)
    b_hh = np.asarray(d["b_hh"], dtype=np.float64)
    out_w = np.asarray(d["out_w"], dtype=np.float64)
    out_b = np.asarray(d["out_b"], dtype=np.float64)
    band_map = np.asarray(d["band_map"], dtype=np.int64)

    n_bins = DEFAULT_N_FFT // 2 + 1
    expected_bands = int(meta["n_bands"])
    hidden = int(meta["hidden"])
    n_features = int(meta["n_features"])
    if band_map.shape != (n_bins,):
        raise ValueError(f"band_map must have {n_bins} bins, got {band_map.shape}")
    if not np.array_equal(band_map, BIN_TO_BAND):
        raise ValueError("band_map does not match this code's canonical mel "
                         "banding; the checkpoint was built for different "
                         "band edges - refuse, don't reinterpret")
    if w_ir.shape != (3 * hidden, n_features) \
            or w_hr.shape != (3 * hidden, hidden) \
            or b_ih.shape != (3 * hidden,) \
            or b_hh.shape != (3 * hidden,) \
            or out_w.shape != (expected_bands, hidden) \
            or out_b.shape != (expected_bands,):
        raise ValueError(f"weight shapes do not match schema: "
                         f"w_ir={w_ir.shape}, w_hr={w_hr.shape}, "
                         f"b_ih={b_ih.shape}, b_hh={b_hh.shape}, "
                         f"out_w={out_w.shape}, out_b={out_b.shape}")

    for name, arr in (("w_ir", w_ir), ("w_hr", w_hr), ("b_ih", b_ih),
                      ("b_hh", b_hh), ("out_w", out_w), ("out_b", out_b)):
        if not np.isfinite(arr).all():
            raise ValueError(f"--gain-model {path} contains non-finite {name}")

    w = LearnedWeights(
        w_ir=w_ir, w_hr=w_hr, b_ih=b_ih, b_hh=b_hh,
        out_w=out_w, out_b=out_b,
        n_features=n_features, hidden=hidden, n_bands=expected_bands)
    return LoadedGainModel(weights=w, meta=meta, path=str(path))


class GainModelHook:
    """Callable gain provider for SpectralSubtraction (the learned step).

    Contract: __call__(mag, noise_mag, frame_index) -> per-bin gains float64
    in [0,1] (length n_bins). Maintains its own GRU hidden state; reset()
    is called by the pipeline on stream reset (deterministic replay).
    """

    def __init__(self, model: LoadedGainModel) -> None:
        self.model = model
        self.hidden: np.ndarray | None = None
        self.frames_processed = 0
        self.reset()

    def reset(self) -> None:
        self.hidden = None
        self.frames_processed = 0

    def __call__(self, mag: np.ndarray, noise_mag: np.ndarray,
                 frame_index: int) -> np.ndarray:
        mag = np.asarray(mag, dtype=np.float64)
        noise_mag = np.asarray(noise_mag, dtype=np.float64)
        n_bins = DEFAULT_N_FFT // 2 + 1
        if mag.shape != (n_bins,) or noise_mag.shape != (n_bins,):
            raise ValueError(f"hook expects ({n_bins},) arrays, got "
                             f"{mag.shape} / {noise_mag.shape}")
        if not np.isfinite(mag).all() or not np.isfinite(noise_mag).all():
            raise ValueError("gain hook received non-finite spectra; refusing")

        f_mix = np.log10(np.maximum(
            band_rms(mag[None, :], BIN_TO_BAND, self.model.weights.n_bands),
            FEATURE_EPS)).ravel()
        f_noise = np.log10(np.maximum(
            band_rms(noise_mag[None, :], BIN_TO_BAND,
                     self.model.weights.n_bands),
            FEATURE_EPS)).ravel()
        feats = np.concatenate([f_mix, f_noise])
        if feats.size != self.model.weights.n_features:
            raise ValueError(
                f"feature size {feats.size} != n_features "
                f"{self.model.weights.n_features}")

        band_gains, self.hidden = gru_forward_numpy(
            self.model.weights, feats[None, :], self.hidden)
        gains = expand_band_gains(band_gains[0], BIN_TO_BAND,
                                  self.model.weights.n_bands)
        self.frames_processed += 1

        # Fail-closed: out-of-range or non-finite gains abort the cell (the
        # caller surfaces this as a benchmark error, never as a "learned"
        # number).
        if not np.isfinite(gains).all():
            raise ValueError("non-finite model gains; refusing frame")
        if bool((gains < -1e-6).any()) or bool((gains > 1 + 1e-6).any()):
            raise ValueError(f"model gains outside [0,1]: min={gains.min():.4f}, "
                             f"max={gains.max():.4f}; refusing frame")
        return np.clip(gains, 0.0, 1.0)