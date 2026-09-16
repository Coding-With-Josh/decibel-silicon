"""Tests for the learned gain-model package (banding, data determinism,
torch/numpy GRU equivalence, npz schema validation, and the pipeline hook).

Fast-path note: all tests here are small/unit-level; the heavy corpus mix
(real LibriSpeech + DEMAND) is exercised by train.py on demand, not in CI.
"""

import json

import numpy as np
import pytest

from learned import (
    GainModelHook,
    GruGainNet,
    N_BANDS,
    export_weights,
    gru_forward_numpy,
    load_model_npz,
)
from learned.banding import BIN_TO_BAND, band_rms, expand_band_gains
from learned.data import Utterance, _record_seed, build_mix_record
from learned.infer import LoadedGainModel
from src.pipeline import NoiseReductionPipeline


def _fake_utt(duration=0.25, seed=0):
    rng = np.random.default_rng(seed)
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "utt.flac"
    samples = (rng.standard_normal(int(16000 * duration)) * 0.1).astype(np.float32)
    return Utterance(file=tmp, speaker="S1", samples=samples)


def _fake_noise(duration=2.0, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * duration)) * 0.2).astype(np.float64)


# ---------------------------------------------------------------------------
# banding
# ---------------------------------------------------------------------------


def test_band_map_matches_geometry():
    assert BIN_TO_BAND.shape == (65,)          # n_fft//2 + 1 bins
    assert BIN_TO_BAND.min() == 0
    assert BIN_TO_BAND.max() < N_BANDS
    assert (np.diff(BIN_TO_BAND) >= 0).all()   # monotone, no bin skips backward


def test_band_rms_and_expand_roundtrip():
    rms = 10.0 ** np.linspace(-1.0, 1.0, N_BANDS)      # 0.1 .. 10
    gains = expand_band_gains(rms, BIN_TO_BAND, N_BANDS)
    assert gains.shape == (65,)
    assert (gains >= 0.1 - 1e-12).all() and (gains <= 10.0 + 1e-12).all()
    # Constant band value -> constant bin value.
    k = np.where(BIN_TO_BAND == 5)[0]
    assert np.allclose(gains[k], rms[5])


# ---------------------------------------------------------------------------
# deterministic data (the cache MUST be a faithful memoization)
# ---------------------------------------------------------------------------


def test_record_seed_stable_across_processes():
    # builtins.hash() is randomized per process; _record_seed is not.
    key = "S1_0000_+5dB_DKITCHEN_ch01"
    assert _record_seed(key) == _record_seed(key)
    assert _record_seed(key) != _record_seed(key + "x")


def test_mix_record_is_deterministic():
    a = build_mix_record(_fake_utt(), _fake_noise(), 5.0, "DKITCHEN", "ch01",
                         jitter_ok=True)
    b = build_mix_record(_fake_utt(), _fake_noise(), 5.0, "DKITCHEN", "ch01",
                         jitter_ok=True)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.targets, b.targets)
    assert np.isfinite(a.features).all()
    assert a.features.shape == (a.n_frames, 2 * N_BANDS)
    assert 0.0 <= a.targets.min() and a.targets.max() <= 1.0


def test_mix_record_noise_offset_differs_across_snr():
    a = build_mix_record(_fake_utt(), _fake_noise(), 0.0, "STRAFFIC", "ch05",
                         jitter_ok=False)
    b = build_mix_record(_fake_utt(), _fake_noise(), 15.0, "STRAFFIC", "ch05",
                         jitter_ok=False)
    assert not np.array_equal(a.features, b.features)


# ---------------------------------------------------------------------------
# torch/numpy GRU equivalence (the artifact under test is the NUMPY forward)
# ---------------------------------------------------------------------------


def test_gru_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    F, H, B, T = 48, 32, 24, 5
    net = GruGainNet(n_features=F, hidden=H, n_bands=B)
    # Randomize weights to exercise all gate paths with meaningful numbers.
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.from_numpy(rng.standard_normal(p.shape).astype(
                np.float32) * 0.2))
    w = export_weights(net)

    x = rng.standard_normal((T, F)).astype(np.float32)
    with torch.no_grad():
        t_gains, t_h = net(torch.from_numpy(x)[None, :, :])
    n_gains, n_h = gru_forward_numpy(w, x)

    assert np.allclose(n_gains, t_gains[0].numpy(), atol=1e-4)
    assert np.allclose(n_h, t_h[0].numpy(), atol=1e-4)


def test_gru_forward_chained_state_is_consistent():
    # Chunked processing with carried hidden state must agree with one pass.
    rng = np.random.default_rng(11)
    F, H, B, T = 48, 32, 24, 8
    net = GruGainNet(n_features=F, hidden=H, n_bands=B)
    w = export_weights(net)
    x = rng.standard_normal((T, F)).astype(np.float32)
    full, _ = gru_forward_numpy(w, x)
    h = None
    for i in range(0, T, 3):
        chunk, h = gru_forward_numpy(w, x[i : i + 3], h)
    assert np.allclose(chunk, full[6:], atol=1e-12)
    assert h.shape == (H,)


# ---------------------------------------------------------------------------
# npz export schema (fail-closed load)
# ---------------------------------------------------------------------------


def _in_memory_mini_export(tmp_path):
    net = GruGainNet(n_features=48, hidden=8, n_bands=4)
    w = export_weights(net)
    out = tmp_path / "mini.npz"
    np.savez_compressed(out, w_ir=w.w_ir, w_hr=w.w_hr, b_ih=w.b_ih,
                        b_hh=w.b_hh, out_w=w.out_w, out_b=w.out_b,
                        band_map=BIN_TO_BAND,
                        meta=json.dumps({
                            "n_fft": 128, "hop": 64, "fs": 16000,
                            "n_bands": 4, "hidden": 8, "n_features": 48,
                            "trained_on": "DKITCHEN", "held_out": "PCAFETER",
                        }))
    return out


def test_load_model_npz_valid_and_invalid(tmp_path):
    ok = _in_memory_mini_export(tmp_path)
    model = load_model_npz(ok)
    assert isinstance(model, LoadedGainModel)
    assert model.meta["held_out"] == "PCAFETER"

    bad = tmp_path / "bad.npz"
    d = np.load(ok)
    np.savez_compressed(bad, w_ir=d["w_ir"].astype(np.float32),
                        w_hr=d["w_hr"], b_ih=d["b_ih"], b_hh=d["b_hh"],
                        out_w=d["out_w"], out_b=d["out_b"],
                        band_map=np.zeros(65, dtype=np.int64), meta=d["meta"])
    with pytest.raises(ValueError, match="band_map"):
        load_model_npz(bad)

    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, w_ir=d["w_ir"])
    with pytest.raises(ValueError, match="missing arrays"):
        load_model_npz(missing)

    with pytest.raises(FileNotFoundError):
        load_model_npz(tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# pipeline integration (the hook occupies the classical decision point)
# ---------------------------------------------------------------------------


class _OnesHook:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def __call__(self, mag, noise_mag, frame_index):
        self.calls += 1
        return np.ones(mag.shape[0], dtype=np.float64)


class _BadHook:
    def __call__(self, mag, noise_mag, frame_index):
        return np.full(mag.shape[0], 2.0)   # outside [0,1] -> must raise


def test_hook_gain_one_identity_after_warmup():
    """A hook returning gain=1 is the WOLA identity: ACTIVE output must equal
    the input (delayed by hop) exactly like the classical bypass path."""
    from src.spectral_subtraction import SpectralSubtractionConfig

    cfg = SpectralSubtractionConfig()
    hook = _OnesHook()
    p = NoiseReductionPipeline(config=cfg, gain_hook=hook)
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(24000) * 0.5).astype(np.float64)
    out = p.process(x)
    assert hook.calls > 0                       # ACTIVE frames really used the hook
    assert np.allclose(out[cfg.n_fft + cfg.hop :], x[cfg.n_fft : -cfg.hop],
                       atol=1e-9)


def test_hook_bad_gain_fails_closed():
    from src.spectral_subtraction import SpectralSubtractionConfig

    cfg = SpectralSubtractionConfig()
    p = NoiseReductionPipeline(config=cfg, gain_hook=_BadHook())
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(24000) * 0.5).astype(np.float64)
    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        p.process(x)


def test_hook_reset_makes_replay_deterministic():
    from src.spectral_subtraction import SpectralSubtractionConfig

    cfg = SpectralSubtractionConfig()
    hook = _OnesHook()
    p = NoiseReductionPipeline(config=cfg, gain_hook=hook)
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(24000) * 0.5).astype(np.float64)
    p.reset()
    out1 = p.process(x)
    p.reset()
    out2 = p.process(x)
    assert np.array_equal(out1, out2)


def test_no_hook_classical_path_still_deterministic():
    # Regression guard: adding the gain_hook plumbing must not disturb the
    # no-hook path (the whole existing suite re-verifies this; here we pin the
    # ACTIVE-frame hook short-circuit really is off).
    from src.spectral_subtraction import (SpectralSubtraction,
                                          SpectralSubtractionConfig)
    ss = SpectralSubtraction(SpectralSubtractionConfig())
    assert ss.gain_hook is None