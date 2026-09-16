"""Tests for the power model (cycle counts and the cited MOPS/mW conversion)."""

import pytest

from src.power_model import (
    BUTTERFLY_ADDS,
    BUTTERFLY_MULTS,
    CV32E40P_MOPS_PER_MW,
    CycleTable,
    estimate_power_mw,
    ops_per_frame,
)


def test_butterfly_count_formula():
    # (N/2)*log2(N) butterflies: N=128 -> 64*7 = 448
    m = ops_per_frame(128, 64)
    assert m.fft_mults == 448 * BUTTERFLY_MULTS
    assert m.fft_adds == 448 * BUTTERFLY_ADDS


def test_fft_ops_scale_n_log_n():
    small = ops_per_frame(64, 32)
    big = ops_per_frame(128, 64)
    # Butterflies = (N/2)*log2(N): 32*6=192 vs 64*7=448 -> ratio 7/3.
    assert big.fft_mults == small.fft_mults * 7 // 3
    assert big.fft_adds == small.fft_adds * 7 // 3


def test_magnitude_bins():
    m = ops_per_frame(128, 64)
    assert m.magnitude_sqrts == 128 // 2 + 1
    assert m.magnitude_mults == 2 * (128 // 2 + 1)


def test_noise_est_ops_only_when_requested():
    m_off = ops_per_frame(128, 64, noise_est=False)
    m_on = ops_per_frame(128, 64, noise_est=True)
    assert m_off.noise_update_mults == 0
    assert m_off.noise_update_adds == 0
    assert m_on.noise_update_mults > 0


def test_subtraction_ops_only_in_active():
    m = ops_per_frame(128, 64, subtract=False)
    assert m.subtract_mults == 0 and m.subtract_divs == 0 and m.subtract_adds == 0
    m2 = ops_per_frame(128, 64, subtract=True)
    assert m2.subtract_divs == 128 // 2 + 1


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        ops_per_frame(100, 50)
    with pytest.raises(ValueError):
        ops_per_frame(128, 30)


def test_conversion_identity_at_cited_efficiency():
    """193 MOPS/mW means 193e6 ops/s consumes 1 mW at the cited operating
    point. The model must reproduce that exactly (this pins the formula)."""
    est = estimate_power_mw(128, 64, 16000)
    cycles_per_sec = est.cycles_per_frame * est.frames_per_second
    assert est.power_mw == pytest.approx(
        cycles_per_sec / (CV32E40P_MOPS_PER_MW * 1e6)
    )


def test_power_monotonic_in_cycles():
    a = estimate_power_mw(64, 32, 16000)
    b = estimate_power_mw(128, 64, 16000)
    c = estimate_power_mw(256, 128, 16000)
    assert a.power_mw < b.power_mw < c.power_mw


def test_estimate_is_labeled_not_measured():
    est = estimate_power_mw(128, 64, 16000)
    assert est.source == "cite"
    assert "not measured" in est.caveat.lower()
    assert "Gautschi" in est.reference


def test_documented_cycle_table_used():
    est = estimate_power_mw(128, 64, 16000, cycle_table=CycleTable(mul=1, div=1, sqrt=1, add=1))
    # With every op cost 1, power == total_ops * frames_per_sec / 193e6.
    m = ops_per_frame(128, 64)
    expected = m.total_operations() * 250.0 / (CV32E40P_MOPS_PER_MW * 1e6)
    assert est.power_mw == pytest.approx(expected)


def test_default_estimate_sane_magnitude():
    """Sanity guard: the default N=128 config estimate stays far below the
    5-15 mW envelope (compute-core-only estimate) - if it ever balloons, the
    model assumptions changed and need re-review."""
    est = estimate_power_mw(128, 64, 16000)
    assert 0.0 < est.power_mw < 1.0


# --------------------------------------------------------------------------
# v2 adaptive ops (VAD-gated tracking + SNR-adaptive alpha_eff)
# --------------------------------------------------------------------------


def test_adaptive_ops_zero_when_flags_off():
    """The v1 manifest must be unchanged: with tracking=False and
    adaptive=False every v2 field is zero (legacy numbers stay identical)."""
    base = ops_per_frame(128, 64, tracking=False, adaptive=False)
    assert base.smooth_mults == 0
    assert base.smooth_adds == 0
    assert base.track_mults == 0
    assert base.track_adds == 0
    assert base.snr_logs == 0
    assert base.alpha_ops == 0


def test_tracking_ops_counted_per_bin():
    """Tracking = 1 mul + 2 add per bin on inactive frames (ACTIVE-only)."""
    m = ops_per_frame(128, 64, tracking=True, adaptive=False)
    bins = 128 // 2 + 1
    assert m.track_mults == bins
    assert m.track_adds == 2 * bins
    # Meters required with tracking on: smoothed-magnitude + energies.
    assert m.smooth_mults == 4 * bins
    assert m.smooth_adds == 3 * bins


def test_adaptive_ops_counted():
    """Adaptive alpha adds one log2 + ~8 alpha-map ops; meters are shared
    with tracking, so they must NOT be double-counted when both are on."""
    adaptive = ops_per_frame(128, 64, tracking=False, adaptive=True)
    assert adaptive.snr_logs == 1
    assert adaptive.alpha_ops == 8
    assert adaptive.smooth_mults > 0  # meters still required for the SNR read


def test_adaptive_overhead_visible_in_power_but_small():
    """Default v2 (tracking+adaptive) must cost more cycles than v1 but stay
    within the same order of magnitude (documented ~+11% overhead: 65-bin
    meters + tracking + one log2; model, not measured)."""
    v1 = estimate_power_mw(128, 64, 16000)
    v2 = estimate_power_mw(128, 64, 16000, tracking=True, adaptive=True)
    delta = v2.cycles_per_frame - v1.cycles_per_frame
    assert v2.cycles_per_frame > v1.cycles_per_frame
    # Exact expected delta: meters (4 mul+3 add per bin), tracking
    # (1 mul+2 add per bin), one log2 (35), alpha map (8).
    bins = 128 // 2 + 1
    expected_delta = (4 * bins) * 4 + (3 * bins) * 1 \
        + bins * 4 + (2 * bins) * 1 + 35 + 8
    assert delta == expected_delta  # model formula stays pinned
    assert v2.cycles_per_frame < v1.cycles_per_frame * 1.15
    assert v2.power_mw < 1.0  # still far below the 5-15 mW envelope


def test_log2_cycle_cost_from_table():
    """The log2 cost comes from the cycle table (default 35 cycles) and is
    counted when adaptive=True only."""
    st = CycleTable()
    assert st.log2 == 35
    m = ops_per_frame(128, 64, adaptive=True)
    assert m.snr_logs == 1
    # All-ones table (same convention as test_documented_cycle_table_used):
    # cycles == total_operations, isolating the adaptive contribution.
    est = estimate_power_mw(128, 64, 16000, adaptive=True,
                            cycle_table=CycleTable(mul=1, div=1, sqrt=1,
                                                   add=1, log2=1))
    m2 = ops_per_frame(128, 64, tracking=False, adaptive=True)
    assert est.cycles_per_frame == pytest.approx(m2.total_operations())