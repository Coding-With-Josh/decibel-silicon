"""Tests for the streaming pipeline: latency probe and ceiling verdicts."""

import numpy as np
import pytest

from src.pipeline import (
    DEFAULT_CEILING_MS,
    NoiseReductionPipeline,
    run_stream,
)
from src.spectral_subtraction import SpectralSubtractionConfig


def make_tone(fs=16000, freq=1000.0, seconds=0.3, amp=0.5):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_probe_measures_hop_group_delay():
    """The structural (group) delay of this WOLA is hop samples = 4 ms at
    defaults. The probe must measure exactly that - this is the number the
    pitch quotes, so it must be deterministic and deliberate, not a
    literature assertion."""
    p = NoiseReductionPipeline()
    delay_ms = p.measure_algorithmic_delay_ms()
    expected_ms = p.config.hop / p.config.fs * 1000.0  # 64/16000 = 4.0 ms
    assert delay_ms == pytest.approx(expected_ms, abs=1e-3)


def test_probe_is_stream_state_independent():
    """The probe must not perturb (or be perturbed by) the caller's stream."""
    p = NoiseReductionPipeline()
    # advance the stream twice; probe uses a temporary internal instance
    p.process_frame(make_tone(seconds=0.05)[: p.config.hop])
    d1 = p.measure_algorithmic_delay_ms()
    p.process_frame(make_tone(seconds=0.05)[: p.config.hop])  # advance stream
    d2 = p.measure_algorithmic_delay_ms()
    assert d1 == pytest.approx(d2, abs=1e-9)
    assert len(p.report().frames) == 2  # untouched by the probes


def test_probe_raises_when_no_response_detectable():
    """Fail-closed: an unreachable threshold must raise (a silent 0.0 ms would
    poison the pitch number)."""
    p = NoiseReductionPipeline()
    with pytest.raises(RuntimeError):
        p.measure_algorithmic_delay_ms(threshold=1e30)


def test_probe_requires_overlap_config():
    cfg = SpectralSubtractionConfig(n_fft=128, hop=128)
    p = NoiseReductionPipeline(config=cfg)
    with pytest.raises(ValueError):
        p.measure_algorithmic_delay_ms()


def test_ceiling_verdicts_default_pass():
    """At defaults (4 ms measured vs 20 ms ceiling) every frame must PASS."""
    p = NoiseReductionPipeline(config=SpectralSubtractionConfig(noise_frames=8))
    x = make_tone(seconds=0.3)
    out, res = p.process_frame(x[: p.config.hop])
    assert res.passed_ceiling is True
    assert res.verdict == "PASS"


def test_ceiling_verdicts_tight_ceiling_fail():
    """A ceiling below the measured delay must fail closed - never 'pass'
    because the number hasn't been measured yet."""
    p = NoiseReductionPipeline(ceiling_ms=1.0, preferred_ms=0.0)  # 1 ms < 4 ms
    x = make_tone(seconds=0.3)
    out, res = p.process_frame(x[: p.config.hop])
    assert res.passed_ceiling is False
    assert res.verdict == "FAIL"


def test_report_aggregation():
    p = NoiseReductionPipeline(ceiling_ms=DEFAULT_CEILING_MS)
    x = make_tone(seconds=0.2)
    report = run_stream(p, x)
    assert report.total_frames == 50  # 0.2s * 16000 / 64 = 50 frames
    assert report.passed_frames == report.total_frames
    assert report.passed_ceiling is True
    assert report.measured_algo_delay_ms == pytest.approx(4.0, abs=1e-3)
    assert report.worst_compute_ms >= 0.0
    assert len(report.summary_lines()) >= 5


def test_run_stream_resets_pipeline():
    p = NoiseReductionPipeline()
    x1 = make_tone(seconds=0.1)
    r1 = run_stream(p, x1)
    n1 = r1.total_frames
    r2 = run_stream(p, x1)
    assert r2.total_frames == n1
    assert r2.frames[0].frame_index == 0  # reset, not continued


def test_budget_flag_is_separate_from_ceiling():
    """passed_budget is the TARGET-only real-time flag (hop/fs); it must never
    influence the ceiling verdict the pitch quotes."""
    p = NoiseReductionPipeline()
    x = make_tone(seconds=0.2)
    report = run_stream(p, x)
    for f in report.frames:
        assert f.passed_ceiling is True  # 4ms <= 20ms
        # The Python prototype is expected to fail the real-time budget; the
        # two flags must be independent.
        assert not (f.passed_budget and not f.passed_ceiling)


def test_invalid_ceiling_rejected():
    with pytest.raises(ValueError):
        NoiseReductionPipeline(ceiling_ms=0.0)
    with pytest.raises(ValueError):
        NoiseReductionPipeline(ceiling_ms=2000.0)
    with pytest.raises(ValueError):
        NoiseReductionPipeline(ceiling_ms=5.0, preferred_ms=20.0)  # ceiling < pref


def test_lazy_delay_measurement_once():
    """The probe runs at most once per stream (lazy, cached per stream)."""
    p = NoiseReductionPipeline()
    p.process_frame(make_tone(seconds=0.02)[: p.config.hop])
    assert p._measured_delay_ms is not None
    cached = p._measured_delay_ms
    p.process_frame(make_tone(seconds=0.02)[: p.config.hop])
    assert p._measured_delay_ms == cached