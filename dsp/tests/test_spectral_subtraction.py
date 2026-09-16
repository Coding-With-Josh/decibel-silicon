"""Tests for the baseline spectral subtraction module."""

import numpy as np
import pytest

from src.spectral_subtraction import (
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    SpectralSubtraction,
    SpectralSubtractionConfig,
    STATE_ACTIVE,
    STATE_NOISE_EST,
)


def make_tone(fs=16000, freq=1000.0, seconds=0.5, amp=0.5):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def make_noisy_tone(fs=16000, freq=1000.0, seconds=0.5, snr_db=5.0, seed=3):
    rng = np.random.default_rng(seed)
    clean = make_tone(fs=fs, freq=freq, seconds=seconds)
    noise = rng.standard_normal(clean.size)
    scale = np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    return clean + scale * noise


# --------------------------------------------------------------------------
# WOLA identity: with the spectral gain forced to 1 (i.e., warm-up
# passthrough), the chain must reconstruct its input exactly **up to the
# structural delay** (hop samples; measured by the impulse probe). The buffer is the newest N
# samples centered on the frame boundary, so interior output equals the input
# delayed by hop: out[s] == x[s - hop]. Quality metrics elsewhere compensate
# this delay; latency is budgeted separately.
# --------------------------------------------------------------------------


def test_identity_reconstruction_during_warmup():
    """During passthrough (noise_est), out[s] == x[s - hop] exactly.

    Only the passthrough region is compared: once ACTIVE begins (frame
    noise_frames), the noise estimate is nonzero (it is built from the signal)
    and subtraction legitimately attenuates - identity no longer applies.
    """
    cfg = SpectralSubtractionConfig()
    ss = SpectralSubtraction(cfg)
    x = make_tone(seconds=0.2)
    out = np.concatenate(
        [ss.process_frame(x[i : i + cfg.hop]) for i in range(0, x.size, cfg.hop)]
    )
    # Frame 0 emits silence (not a full buffer yet -> Phase 2 no-op).
    s_start = cfg.n_fft + cfg.hop                       # first fully-interior block
    s_end = cfg.noise_frames * cfg.hop                  # ACTIVE begins here
    assert np.allclose(out[s_start:s_end], x[cfg.n_fft : s_end - cfg.hop],
                       atol=1e-6)


def test_identity_reconstruction_delay_is_hop():
    """With gain forced to 1.0 (alpha=0, floor=1), the WHOLE stream must
    reconstruct exactly - and the true relation is the hop-sample structural
    delay, pinned here for a non-tonal signal (a time-aligned identity would
    fail)."""
    cfg = SpectralSubtractionConfig(noise_frames=4, alpha=0.0, floor=1.0)
    ss = SpectralSubtraction(cfg)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(cfg.n_fft * 8)
    out = np.concatenate(
        [ss.process_frame(x[i : i + cfg.hop]) for i in range(0, x.size, cfg.hop)]
    )
    assert ss.state == STATE_ACTIVE
    # out[s] == x[s - hop] for the fully-interior region.
    assert np.allclose(out[cfg.n_fft + cfg.hop :], x[cfg.n_fft : -cfg.hop],
                       atol=1e-5)
    # The un-delayed claim must FAIL for this signal (guards the test).
    assert not np.allclose(out[cfg.n_fft:], x[cfg.n_fft:], atol=1e-3)


def test_identity_reconstruction_after_warmup_with_zero_subtraction():
    # alpha 0 + floor 1 -> gain exactly 1.0 in ACTIVE: identity must hold.
    cfg = SpectralSubtractionConfig(alpha=0.0, floor=1.0, noise_frames=8,
                                    n_fft=128, hop=64)
    ss = SpectralSubtraction(cfg)
    x = make_tone(seconds=0.4)
    out = np.concatenate(
        [ss.process_frame(x[i : i + cfg.hop]) for i in range(0, x.size, cfg.hop)]
    )
    assert ss.state == STATE_ACTIVE
    assert np.allclose(out[cfg.n_fft + cfg.hop :], x[cfg.n_fft : -cfg.hop],
                       atol=1e-4)


def test_state_machine_transitions():
    cfg = SpectralSubtractionConfig(noise_frames=4)
    ss = SpectralSubtraction(cfg)
    assert ss.state == "cold_start"
    ss.process_frame(np.zeros(cfg.hop))
    assert ss.state == "cold_start"  # only one full frame consumed so far
    ss.process_frame(np.zeros(cfg.hop))
    assert ss.state == STATE_NOISE_EST  # first full frame processed
    # Accumulation frames k=2..3 are still NOISE_EST (noise_frames-2 more),
    # the noise_frames'th accumulation (k=4) flips to ACTIVE.
    for _ in range(cfg.noise_frames - 2):
        ss.process_frame(np.zeros(cfg.hop))
        assert ss.state == STATE_NOISE_EST
    ss.process_frame(np.zeros(cfg.hop))
    assert ss.state == STATE_ACTIVE


def test_snr_improvement_on_noisy_tone():
    """The baseline must improve SNR on a stationary noisy tone.

    Uses a noise-only leader for the initial noise estimate (the baseline's
    contract: estimate = first frames; real use = quiet-period estimate), then
    measures segmental SNR on the ACTIVE (steady) region.
    """
    cfg = SpectralSubtractionConfig(noise_frames=10)
    ss = SpectralSubtraction(cfg)
    x = make_tone(seconds=0.6)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(x.size)
    scale = np.sqrt(np.mean(x**2) / (np.mean(noise**2) * 10 ** (5 / 10)))
    noisy = x + scale * noise

    # Leader: pure noise at the SAME level as the mixture's noise component
    # (not the mixture RMS - that would over-estimate the noise by
    # sqrt(1 + 10^(SNR/10)) and push the v2 VAD/SNR meters off by the same
    # amount, silently breaking the adaptive path under test).
    lead_len = (cfg.noise_frames + 1) * cfg.hop
    lead = (scale * noise)[:lead_len]
    stream = np.concatenate([lead[:lead_len], noisy])

    out = np.concatenate(
        [ss.process_frame(stream[i : i + cfg.hop])
         for i in range(0, stream.size, cfg.hop)]
    )
    assert ss.state == STATE_ACTIVE

    def snr(clean, est):
        num = np.mean(clean**2)
        den = np.mean((clean - est) ** 2) + 1e-12
        return 10 * np.log10(num / den)

    # Speech region starts after the leader; output is delayed by hop
    # (structural delay); align before comparing, as the benchmark does for
    # quality metrics.
    skip = lead_len
    out_al = out[skip + cfg.hop:]
    x_al = x[: out_al.size]
    nz_al = noisy[: out_al.size]
    snr_in = snr(x_al, nz_al)
    snr_out = snr(x_al, out_al)
    assert snr_out > snr_in, f"expected improvement, got in={snr_in:.1f} out={snr_out:.1f} dB"


def test_spectral_floor_prevents_runaway_attenuation():
    """With a floor, the gain can never go below floor, so output energy
    can't collapse to ~0 even under aggressive subtraction."""
    cfg = SpectralSubtractionConfig(alpha=5.0, floor=0.05, noise_frames=8)
    ss = SpectralSubtraction(cfg)
    x = make_tone(seconds=0.3)
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(x.size)
    scale = np.sqrt(np.mean(x**2) / (np.mean(noise**2) * 10 ** (-5 / 10)))
    noisy = x + scale * noise
    out = np.concatenate(
        [ss.process_frame(noisy[i : i + cfg.hop]) for i in range(0, noisy.size, cfg.hop)]
    )
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 0.01  # output didn't collapse


def test_nan_input_rejected():
    cfg = SpectralSubtractionConfig()
    ss = SpectralSubtraction(cfg)
    bad = np.full(cfg.hop, np.nan)
    with pytest.raises(ValueError):
        ss.process_frame(bad)


def test_wrong_frame_size_rejected():
    ss = SpectralSubtraction()
    with pytest.raises(ValueError):
        ss.process_frame(np.zeros(DEFAULT_HOP + 1))


def test_reset_replays_identically():
    cfg = SpectralSubtractionConfig()
    x = make_tone(seconds=0.3)
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(x.size)
    scale = np.sqrt(np.mean(x**2) / (np.mean(noise**2) * 10 ** (0 / 10)))
    noisy = x + scale * noise

    def run():
        ss = SpectralSubtraction(cfg)
        return np.concatenate(
            [ss.process_frame(noisy[i : i + cfg.hop])
             for i in range(0, noisy.size, cfg.hop)]
        )

    a = run()
    b = run()
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Config validation (fail-closed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fft": 100},          # not a power of two
        {"n_fft": 64, "hop": 24},  # hop does not divide n_fft
        {"hop": 0},              # hop zero
        {"hop": 200, "n_fft": 128},  # hop > n_fft
        {"alpha": -0.5},         # alpha negative (0.0 IS allowed: no subtr.)
        {"alpha": 20.0},         # alpha too large
        {"floor": 1.5},          # floor >= 1
        {"noise_frames": 0},     # must be >= 1
        {"fs": -1},              # fs must be positive
        # v2 adaptive config (fail-closed: every bound enforced at the door).
        {"noise_update_mu": 1.0},   # mu == 1: tracker would freeze est at sample
        {"noise_update_mu": 1.5},   # mu > 1: unstable
        {"noise_update_mu": -0.1},  # mu < 0: unbounded
        {"vad_threshold_db": 0.0},  # VAD threshold must be positive (0 = always on)
        {"vad_threshold_db": -3.0},
        {"vad_hangover_frames": -1},  # hangover must be non-negative
        {"alpha_snr_slope": 0.6},     # slope clamp domain (0.5 max)
        {"alpha_snr_slope": -0.6},
        {"alpha_min": 3.5, "alpha_max": 3.0},   # min > max
        {"alpha_max": 12.0},          # alpha domain top (10 max) - driven by
                                      # identity-guard: alpha<=0 or floor>=1
                                      # bypasses alpha_eff entirely.
    ],
)
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        SpectralSubtractionConfig(**kwargs)