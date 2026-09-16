"""Tests for the adaptive revision 2 behavior (v2).

Revision 2 adds two mechanisms to the ACTIVE state (see
src/spectral_subtraction.py module docstring):
  1. VAD-gated recursive noise tracking (fix: AM noise outruns the static
     estimate).
  2. SNR-adaptive over-subtraction factor alpha_eff (fix: over-subtraction /
     inter-harmonic speech attenuation).

These tests assert the MECHANISM (does the estimate move, does alpha_eff
move, does the VAD gate actually gate) - they do not re-assert the v1
behavior tests, and a regression in the adaptive logic specifically would
fail here even if every v1 test still passed.

SIGNAL WIRING CONTRACT (matches benchmark.build_test_stream): the noise-only
leader MUST be the same scaled noise instance used inside the speech region.
Sizing the leader to the mixture RMS (tone+noise) instead over-estimates the
noise by 10*log10(1 + 10^(SNR/10)) dB and silently breaks every VAD/SNR
assertion below - that is a test bug, not an algorithm bug.
"""

import numpy as np
import pytest

from src.spectral_subtraction import (
    DEFAULT_ALPHA,
    SpectralSubtraction,
    SpectralSubtractionConfig,
    STATE_ACTIVE,
)


def make_tone(fs=16000, freq=1000.0, seconds=0.2, amp=0.5):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def noise(level: float, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return level * x / (np.sqrt(np.mean(x**2)) or 1.0)


def build_stream(cfg: SpectralSubtractionConfig, snr_db: float, *,
                 tone_seconds: float = 1.0, seed: int = 50,
                 tone_amp: float = 0.5) -> np.ndarray:
    """Leader + speech-region stream with the leader at the NOISE-COMPONENT
    level, exactly like build_test_stream (noise attenuated once for snr_db,
    leader = the first frames of that same scaled noise)."""
    hop = cfg.hop
    tone = make_tone(seconds=tone_seconds, amp=tone_amp)
    lead_len = (cfg.noise_frames + 1) * hop
    rng = np.random.default_rng(seed)
    noise_ = rng.standard_normal(lead_len + tone.size)
    scale = np.sqrt(np.mean(tone**2) / (np.mean(noise_**2) * 10 ** (snr_db / 10.0)))
    scaled = scale * noise_
    lead = scaled[:lead_len]
    mixture = tone + scaled[lead_len : lead_len + tone.size]
    return np.concatenate([lead, mixture])


def run(ss: SpectralSubtraction, stream: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [ss.process_frame(stream[i : i + ss.config.hop])
         for i in range(0, stream.size, ss.config.hop)]
    )


def first_alpha_eff_and_energy(ss: SpectralSubtraction,
                               stream: np.ndarray):
    """Run the stream frame-by-frame; return (e_start, final_energy) where
    e_start is the noise-estimate energy at the first ACTIVE frame."""
    hop = ss.config.hop
    e_start = None
    for i in range(0, stream.size, hop):
        ss.process_frame(stream[i : i + hop])
        if e_start is None and ss.state == STATE_ACTIVE:
            e_start = float(np.sum(ss.noise_magnitude**2))
    assert ss.state == STATE_ACTIVE
    return e_start, float(np.sum(ss.noise_magnitude**2))


# --------------------------------------------------------------------------
# VAD-gated noise tracking: the estimate must CHANGE over time when the
# noise level changes, and must NOT follow speech.
# --------------------------------------------------------------------------


def test_noise_estimate_tracks_quiet_region():
    """Leader at level A, then quieter noise B persists -> the estimate must
    decay toward B on the VAD-inactive (noise-only) frames. This is the exact
    'modulated noise outrunning the static estimate' failure the v1 baseline
    had, measured at the mechanism level."""
    cfg = SpectralSubtractionConfig(noise_frames=8, noise_update_mu=0.3,
                                    noise_tracking=True)  # mechanism explicit
    ss = SpectralSubtraction(cfg)
    hop = cfg.hop
    lead = noise(0.5, (cfg.noise_frames + 1) * hop, seed=1)   # level A
    quiet = noise(0.125, 40 * hop, seed=2)                    # level B = A/4
    e_start, e_end = first_alpha_eff_and_energy(ss, np.concatenate([lead, quiet]))
    assert e_start is not None
    # Downward-only gate fires while the frame is quieter than the estimate,
    # then STOPS once converged (anti-cascade). 12-25 updates across 40 frames
    # with mu=0.3 => 98%+ converged; the count range pins "fired, then stopped".
    assert 8 <= ss.noise_update_count <= 25, (
        "quiet frames must trigger downward updates then stop, got "
        f"{ss.noise_update_count}")
    # Estimate must move clearly toward B (statically it would stay at A).
    # E_B/E_A = (0.125/0.5)^2 = 1/16; assert it crossed half-way AND did not
    # collapse below the B level (moving toward, not overshooting).
    assert e_end < e_start * 0.5, f"estimate frozen: {e_end} vs {e_start}"
    assert e_end > e_start * 0.03, f"estimate overshot below B level: {e_end}"


def test_noise_estimate_does_not_follow_speech():
    """Leader level A, then tone at 5 dB SNR -> VAD holds the estimate at the
    leader value; speech frames must never leak into the noise estimate."""
    cfg = SpectralSubtractionConfig(noise_frames=8, noise_update_mu=0.3,
                                    noise_tracking=True)  # mechanism explicit
    ss = SpectralSubtraction(cfg)
    e_start, _ = first_alpha_eff_and_energy(
        ss, build_stream(cfg, 5.0, seed=3))
    # Speech frames at +5 dB sit ~6.2 dB above the VAD threshold -> ~no updates.
    assert ss.noise_update_count <= 2, (
        "speech frames must not trigger noise updates "
        f"(updates={ss.noise_update_count})"
    )
    # Estimate energy must stay at the leader level (never chased speech).
    e_end = float(np.sum(ss.noise_magnitude**2))
    assert e_end == pytest.approx(e_start, rel=0.05)


def test_tracker_never_follows_louder_noise():
    """The downward-only gate: a noise level ABOVE the estimate must NOT be
    tracked (no upward drift toward speech/whatever is louder). The estimate
    can only descend. This is the property that terminates the measure
    cascade where a not-speech-only gate chased a steady tone."""
    cfg = SpectralSubtractionConfig(noise_frames=8, noise_update_mu=0.3,
                                    noise_tracking=True)  # mechanism explicit
    ss = SpectralSubtraction(cfg)
    hop = cfg.hop
    lead = noise(0.5, (cfg.noise_frames + 1) * hop, seed=8)  # level A
    loud = noise(2.0, 40 * hop, seed=9)                      # level 4A (+12 dB)
    e_start, e_end = first_alpha_eff_and_energy(ss, np.concatenate([lead, loud]))
    assert ss.noise_update_count == 0, ("tracker must never fire upward, got "
                                        f"{ss.noise_update_count} updates")
    assert e_end == pytest.approx(e_start, rel=1e-3)  # estimate unchanged


def test_vad_hangover_holds_estimate_through_speech_tail():
    """A speech burst followed by brief low-energy frames must not update the
    estimate until the hangover expires (trailing speech/plosive energy must
    not taint the noise estimate)."""
    cfg = SpectralSubtractionConfig(noise_frames=8, noise_update_mu=0.3,
                                    vad_hangover_frames=3,
                                    noise_tracking=True)  # mechanism explicit
    ss = SpectralSubtraction(cfg)
    hop = cfg.hop
    lead = noise(0.5, (cfg.noise_frames + 1) * hop, seed=5)
    burst = make_tone(seconds=0.2, amp=0.5) + noise(0.2, int(0.2 * 16000), seed=6)
    gap = noise(0.125, 16 * hop, seed=7)  # 16 frames 12 dB QUIETER (trackable
    #                                    noise): hangover holds the head, then
    #                                    the tail is free to track
    stream = np.concatenate([lead, burst, gap])
    run(ss, stream)
    # Hangover must block the first frames of the tail, then tracking resumes.
    # 16 tail frames: if NOTHING were blocked we'd see 16 updates; hangover
    # blocks >= 2, and the estimate converges so some tail frames stop firing.
    assert 2 <= ss.noise_update_count <= 13, (
        "hangover must block the tail, then resume updates; "
        f"got updates={ss.noise_update_count}"
    )


# --------------------------------------------------------------------------
# SNR-adaptive alpha_eff
# --------------------------------------------------------------------------


def _alpha_eff_at_snr(snr_db: float, *, noise_update_mu: float = 0.0,
                      slope: float = 0.1, alpha_max: float = 3.0):
    """Stream at the given global SNR (tracking OFF to isolate alpha_eff;
    VAD-gated tracking is covered separately) and return the final alpha_eff.
    Leader == noise-component level (see module docstring contract)."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    noise_tracking=False,
                                    adaptive_alpha=True,
                                    noise_update_mu=noise_update_mu,
                                    alpha_snr_slope=slope,
                                    alpha_max=alpha_max)
    ss = SpectralSubtraction(cfg)
    run(ss, build_stream(cfg, snr_db, seed=10))
    assert ss.state == STATE_ACTIVE
    return ss.last_alpha_eff


def test_alpha_eff_gentle_at_both_snr_extremes():
    """The tent map: alpha_eff peaks at mid SNR and relaxes at BOTH extremes.
    Low-SNR relaxation fixes over-subtraction at 0 dB (STOI collapse); high-SNR
    relaxation fixes musical-noise on clean-ish input (alpha ~ 1 there was the
    best measured v1 cell)."""
    mid = _alpha_eff_at_snr(10.0)
    low = _alpha_eff_at_snr(-5.0)
    hi = _alpha_eff_at_snr(20.0)
    # Peak location is the noise-estimate-bounded mid cell; measured estimate
    # deviates from the ideal 10 dB by windowing, so assert the SHAPE with
    # generous tolerances, and the strict relaxations at the extremes.
    assert low < mid, f"expected low-SNR alpha below mid, got low={low} mid={mid}"
    assert hi < mid, f"expected high-SNR alpha below mid, got hi={hi} mid={mid}"
    assert low < DEFAULT_ALPHA - 0.2, f"low-SNR alpha_eff must relax, got {low}"
    assert hi < DEFAULT_ALPHA - 0.2, f"high-SNR alpha_eff must relax, got {hi}"


def test_alpha_eff_clamped_to_config_bounds():
    """Even an extreme SNR estimate must never push alpha_eff outside
    [alpha_min, alpha_max] (fail-closed gain domain). The meter's floor is
    0 dB (an energy ratio of mixture/noise can never read below ~0 dB), so
    the practical cells are: -30 dB real -> measured ~0 dB -> tent floor;
    +40 dB real -> measured >> ref -> ALSO floor; the peak cell (10 dB real)
    drives alpha_eff to the ceiling when alpha_max < alpha."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    noise_tracking=False,
                                    adaptive_alpha=True,
                                    alpha_max=1.5)   # tent peak > ceiling
    # -30 dB cell: alpha_eff must not fall below alpha_min.
    assert _alpha_eff_at_snr(-30.0, alpha_max=1.5) == pytest.approx(cfg.alpha_min)
    # Peak cell: must not exceed alpha_max (clip engages on the tent top).
    assert _alpha_eff_at_snr(10.0, alpha_max=1.5) == pytest.approx(cfg.alpha_max)
    # +40 dB cell: also relaxed toward alpha_min on the high side.
    assert _alpha_eff_at_snr(40.0, alpha_max=1.5) == pytest.approx(cfg.alpha_min)


# --------------------------------------------------------------------------
# Identity-test bypass + legacy static path
# --------------------------------------------------------------------------


def test_identity_guard_holds_with_adaptive_defaults():
    """The WOLA identity modes (alpha=0 / floor=1) must stay exact even when
    the adaptive defaults are on; the adaptive mapping is bypassed outright."""
    cfg = SpectralSubtractionConfig(noise_frames=4, alpha=0.0, floor=1.0)
    ss = SpectralSubtraction(cfg)
    x = np.random.default_rng(21).standard_normal(cfg.n_fft * 8)
    out = run(ss, x)
    assert ss.last_alpha_eff == pytest.approx(0.0)
    assert ss.state == STATE_ACTIVE
    assert np.allclose(out[cfg.n_fft + cfg.hop :], x[cfg.n_fft : -cfg.hop],
                       atol=1e-5)


def test_legacy_static_path_requires_flags_off():
    """With noise_tracking=False AND adaptive_alpha=False the estimate must
    stay frozen at the leader value even when the noise level changes - this
    is the v1 behavior the v2 defaults replace (kept as an escape hatch and
    for honest A/B)."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    noise_tracking=False,
                                    adaptive_alpha=False)
    ss = SpectralSubtraction(cfg)
    hop = cfg.hop
    lead = noise(0.5, (cfg.noise_frames + 1) * hop, seed=30)
    quiet = noise(0.125, 40 * hop, seed=31)
    stream = np.concatenate([lead, quiet])
    run(ss, stream)
    assert ss.noise_update_count == 0
    assert ss.last_alpha_eff == pytest.approx(DEFAULT_ALPHA)


def test_adaptive_state_deterministic_on_replay():
    """Identical input after reset() -> identical output, with the full
    adaptive state machine active (determinism contract holds for v2)."""
    cfg = SpectralSubtractionConfig()
    stream = build_stream(cfg, 3.0, seed=40)

    def run_once():
        s = SpectralSubtraction(cfg)
        return run(s, stream)

    a = run_once()
    b = run_once()
    assert np.array_equal(a, b)


def test_all_zero_stream_stays_finite_and_empty_estimate():
    """Fail-closed: an all-zero stream exercises the eps-guarded VAD/SNR
    ratios; the chain must never emit NaN. Silence sits AT the estimate level
    (ratio ~ 0 dB), so the downward-only tracker correctly does NOT fire; the
    estimate must stay exactly zero and alpha_eff must stay finite."""
    cfg = SpectralSubtractionConfig(noise_tracking=True)  # tracker explicit
    ss = SpectralSubtraction(cfg)
    n = cfg.n_fft * 6
    stream = np.zeros(n)
    out = run(ss, stream)
    assert np.isfinite(out).all()
    assert np.isfinite(ss.last_alpha_eff)
    assert float(np.sum(ss.noise_magnitude**2)) == 0.0
    assert ss.noise_update_count == 0  # nothing quieter than the estimate


# --------------------------------------------------------------------------
# Revision 2.1: high-SNR bypass (identity above threshold)
# --------------------------------------------------------------------------


def test_high_snr_bypass_identity_above_threshold():
    """At 20 dB global SNR with threshold 10 dB, virtually every ACTIVE frame
    reads above the threshold: subtraction is skipped and the stream comes
    back EXACTLY (the verified WOLA identity). This is the fix for the
    residual 20 dB segSNR loss. The tone's final frames (energy decays toward
    zero) legitimately read BELOW the threshold and subtract - the identity
    assertion is restricted to the interior where bypass is in force."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    high_snr_bypass_db=10.0)
    ss = SpectralSubtraction(cfg)
    stream = build_stream(cfg, 20.0, seed=60)
    out = run(ss, stream)
    assert ss.state == STATE_ACTIVE
    assert ss.active_frames > 0
    assert ss.bypass_count > 0
    # Allow the end-of-tone frames (energy falls below the threshold there);
    # the mechanism claim is "overwhelmingly bypassed while the tone is loud".
    assert ss.bypass_count >= ss.active_frames - 8, (
        f"expected ~all ACTIVE frames to bypass at 20 dB / thresh 10, "
        f"got bypass={ss.bypass_count} active={ss.active_frames}")
    # Identity on the interior AFTER the meter converges (the first ~3 ACTIVE
    # frames read the windowed leader/mixture blend, legitimately subtract
    # during warm-up, and are excluded): out[j] == in[j - hop] wherever
    # gain == 1. The window stays well inside the strong-tone region.
    w = 12000
    start = (cfg.noise_frames + 6) * cfg.hop   # ~6 frames after ACTIVE starts
    assert np.allclose(out[start : start + w],
                       stream[start - cfg.hop : start - cfg.hop + w],
                       atol=1e-5)


def test_high_snr_bypass_not_fired_at_low_snr():
    """At 0 dB global SNR the meter never reaches the 15 dB threshold:
    the bypass must NOT fire and subtraction keeps running (output differs
    from identity). This pins the fail-safe direction: bypass is a quality
    decision, and a noisy frame must never pass through unprocessed."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    high_snr_bypass_db=15.0)
    ss = SpectralSubtraction(cfg)
    stream = build_stream(cfg, 0.0, seed=61)
    out = run(ss, stream)
    assert ss.bypass_count == 0, f"expected no bypass at 0 dB, got {ss.bypass_count}"
    assert ss.active_frames > 0
    # Subtraction must have actually processed the speech region.
    assert not np.allclose(out[cfg.n_fft + cfg.hop :],
                           stream[cfg.n_fft : -cfg.hop], atol=1e-3)


def test_bypass_threshold_honored_no_hysteresis():
    """A threshold ABOVE the meter (20 dB stream, 40 dB threshold) never
    fires: the decision is a strict comparison on the smoothed estimate, no
    default-bypass, no hysteresis."""
    cfg = SpectralSubtractionConfig(noise_frames=8,
                                    high_snr_bypass_db=40.0)
    ss = SpectralSubtraction(cfg)
    stream = build_stream(cfg, 20.0, seed=62)
    run(ss, stream)
    assert ss.bypass_count == 0


def test_bypass_disabled_by_default():
    """high_snr_bypass_db=None (default) -> the decision path never runs;
    the counter stays zero even on a clean 20 dB stream."""
    cfg = SpectralSubtractionConfig()  # defaults: bypass None
    ss = SpectralSubtraction(cfg)
    stream = build_stream(cfg, 20.0, seed=63)
    run(ss, stream)
    assert ss.active_frames > 0
    assert ss.bypass_count == 0


def test_bypass_invalid_config_rejected():
    """Fail-closed config validation: the threshold must be None or a finite
    value > 0 dB. <= 0 would classify near-silence as clean (quality open),
    NaN/Inf would poison the comparison."""
    for bad in (-5.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SpectralSubtractionConfig(high_snr_bypass_db=bad)
    # The valid sentinel still works alongside defaults.
    assert SpectralSubtractionConfig(
        high_snr_bypass_db=12.0).high_snr_bypass_db == 12.0


def test_all_zero_stream_with_bypass_fail_closed():
    """All-zero stream + bypass enabled: the eps-guarded meter reads ~0 dB,
    far below any sane threshold -> no bypass (silence must not be classified
    as clean), output stays finite, counter stays zero."""
    cfg = SpectralSubtractionConfig(noise_tracking=True,
                                    high_snr_bypass_db=15.0)
    ss = SpectralSubtraction(cfg)
    n = cfg.n_fft * 6
    stream = np.zeros(n)
    out = run(ss, stream)
    assert np.isfinite(out).all()
    assert np.isfinite(ss.last_alpha_eff)
    assert ss.bypass_count == 0