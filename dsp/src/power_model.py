"""Compute-cycle and power estimation model for the DSP chain.

Model inputs (all documented assumptions, none measured on silicon):
  - Operation counts per frame, derived from the exact structure of
    spectral_subtraction.SpectralSubtraction (the baseline algorithm).
  - Per-operation cycle costs for a representative low-power RISC-V core,
    CV32E40P (OpenHW Group, formerly PULP RI5CY), configured RV32IMC with no
    FPU and no PULP DSP extensions (extensions are a future optimization).
  - Power conversion: cited peak energy efficiency for that core:
      193 MOPS/mW in 28 nm FD-SOI at 40 MHz / ~1 mW
    Gautschi, M., et al., "A Near-Threshold RISC-V Core With DSP Extensions
    for Scalable IoT Endpoint Devices," IEEE Trans. VLSI Syst. 25(10):
    2700-2713, Oct. 2017. doi:10.1109/TVLSI.2017.2654506 (arXiv:1608.08376).

Cross-check: the same paper reports 12.26-33.8 uW/MHz dynamic power (65 nm,
1.08 V, various configurations); we report both views.

OUTPUT LABELING (Phase 1 mediation): every estimate returned by this module
is tagged ``source="cite"`` plus an explicit disclaimer. These are MODEL
ESTIMATES, not measurements; the only measured quantities in this repo come
from pipeline/benchmark runs and firmware host tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Cited constant(s)
# --------------------------------------------------------------------------
# Peak energy efficiency of CV32E40P (RI5CY) in 28 nm FD-SOI @ 40 MHz, ~1 mW.
# Source: Gautschi et al. 2017, IEEE TVLSI 25(10):2700-2713.
# DOI 10.1109/TVLSI.2017.2654506.
CV32E40P_MOPS_PER_MW = 193.0  # mega-operations per second per milliwatt
_CV32E40P_MOPS_PER_MW_REF = (
    "Gautschi et al. 2017, IEEE Trans. VLSI Syst. 25(10):2700-2713, "
    "doi:10.1109/TVLSI.2017.2654506, arXiv:1608.08376"
)
# Dynamic power per MHz (65 nm, 1.08 V) from the same paper, min..max.
_CV32E40P_UW_PER_MHZ_RANGE = (12.26, 33.8)
_CV32E40P_UW_PER_MHZ_REF = _CV32E40P_MOPS_PER_MW_REF

# --------------------------------------------------------------------------
# Per-operation cycle costs (ASSUMED, documented - not measured).
# Target: CV32E40P-class RV32IMC core, no FPU.
#   - add/sub/compare/shift/and/or: 1 cycle (in-order 4-stage ALU)
#   - 32x32 multiply: 4 cycles (M extension, 64-bit result)
#   - 32/32 divide: 10 cycles (M extension)
#   - fixed-point square root (Newton iteration): ~20 cycles (estimates)
# These are engineering estimates to be replaced by cycle-accurate profiling
# or silicon measurement; they feed the pitch number, so they must stay
# labeled as assumptions until then.
# --------------------------------------------------------------------------
CYCLE_COST_ADD = 1
CYCLE_COST_MUL = 4
CYCLE_COST_DIV = 10
CYCLE_COST_SQRT = 20
# Fixed-point log2 of a 32-bit ratio (exponent extraction + linear
# interpolation - a common cheap approximation on cores without an FPU).
# ASSUMED cost, documented, to be replaced by cycle-accurate profiling.
CYCLE_COST_LOG2 = 35

# Radix-2 complex butterfly cost in scalar RISC-V ops (RV32IMC, no SIMD):
# one complex multiply (4 real mults + 2 real adds) + 2 complex add/sub
# (4 adds + 2 subs) = 4 mults + 8 add/sub.
BUTTERFLY_MULTS = 4
BUTTERFLY_ADDS = 8


@dataclass
class CycleTable:
    """Documented per-op cycle costs (all assumptions; see module docstring)."""

    add: int = CYCLE_COST_ADD
    mul: int = CYCLE_COST_MUL
    div: int = CYCLE_COST_DIV
    sqrt: int = CYCLE_COST_SQRT
    log2: int = CYCLE_COST_LOG2


@dataclass
class OpsPerFrame:
    """Operation counts per frame, derived from the baseline algorithm.

    Mirrors the documented structure of SpectralSubtraction.process_frame:
        window-multiply      : N mults (analysis window)
        forward FFT          : (N/2)*log2(N) butterflies x (4 mul + 8 add)
        magnitude            : (N/2+1) bins x (2 mul + 1 sqrt)
        noise estimator      : (N/2+1) bins x (2 mul + 2 add), NOISE_EST only
        subtraction          : (N/2+1) bins x (2 mul + 1 div + 2 add), ACTIVE
        reconstruction       : (N/2+1) bins x (2 mul)   (real gain * complex)
        inverse FFT          : same as forward FFT
        synthesis window     : N mults
        overlap-add + norm   : N adds + hop mults
    v2 adaptive (all ACTIVE-only; gated by the tracking/adaptive flags):
        smoothed-magnitude + VAD/SNR energy meters
smooth_mults     : bins x 4 (sm blend 2 mul + energy sqrs 2 mul)
        smooth_adds      : bins x 3 (sm blend 1 add + energy sums 2 add)
        noise tracking   : bins x (1 mul + 2 add) on inactive frames
        SNR estimate     : 1 fixed-point log2 + linear alpha map (~8 ops)

    These counts are assumptions tied to the algorithm as implemented. If the
    algorithm changes, this model changes with it (never silently).
    """

    n_fft: int
    hop: int
    window_mults: int = 0
    fft_mults: int = 0
    fft_adds: int = 0
    magnitude_mults: int = 0
    magnitude_sqrts: int = 0
    noise_update_mults: int = 0
    noise_update_adds: int = 0
    subtract_mults: int = 0
    subtract_divs: int = 0
    subtract_adds: int = 0
    reconstruct_mults: int = 0
    ola_adds: int = 0
    ola_norm_mults: int = 0
    # v2 adaptive manifest (see module docstring / noise-reduction.md).
    smooth_mults: int = 0
    smooth_adds: int = 0
    track_mults: int = 0
    track_adds: int = 0
    snr_logs: int = 0
    alpha_ops: int = 0

    def total_operations(self) -> int:
        return (
            self.window_mults
            + self.fft_mults
            + self.fft_adds
            + self.magnitude_mults
            + self.magnitude_sqrts
            + self.noise_update_mults
            + self.noise_update_adds
            + self.subtract_mults
            + self.subtract_divs
            + self.subtract_adds
            + self.reconstruct_mults
            + self.ola_adds
            + self.ola_norm_mults
            + self.smooth_mults
            + self.smooth_adds
            + self.track_mults
            + self.track_adds
            + self.snr_logs
            + self.alpha_ops
        )


def ops_per_frame(n_fft: int, hop: int, *, noise_est: bool = False,
                  subtract: bool = True, tracking: bool = False,
                  adaptive: bool = False) -> OpsPerFrame:
    """Build the per-frame operation manifest for config (n_fft, hop).

    noise_est=True counts the noise-estimator update ops (active only while
    state == NOISE_EST); subtract=False yields the passthrough warm-up count.
    tracking=True / adaptive=True count the v2 ACTIVE-only meters (VAD-gated
    noise update, SNR estimate + alpha map) exactly as implemented in
    spectral_subtraction.process_frame. The v1 manifest (defaults) is
    unchanged - these counts are additive, never silently merged.
    """
    if n_fft < 8 or n_fft & (n_fft - 1):
        raise ValueError(f"n_fft must be a power of two >= 8, got {n_fft}")
    if not (0 < hop <= n_fft) or n_fft % hop:
        raise ValueError(f"hop must divide n_fft, got hop={hop}, n_fft={n_fft}")

    butterflies = (n_fft // 2) * int(np_log2(n_fft))  # (N/2) * log2(N)
    bins = n_fft // 2 + 1

    # v2 ACTIVE-only meters (see spectral_subtraction.process_frame):
    #   smoothing  : sm = (1-eta)*sm + eta*mag          -> 2 mul + 1 add/bin
    #   energy_sig : sum(sm^2)                          -> 1 mul + 1 add/bin
    #   energy_noi : sum(noise^2)                       -> 1 mul + 1 add/bin
    #   tracking   : noise += mu*(mag - noise)          -> 1 mul + 2 add/bin
    #   snr log2   : one fixed-point log2 of the ratio  -> CYCLE_COST_LOG2
    #   alpha map  : linear map + 2 clamps + store      -> ~8 ops (assumed)
    smooth_mults = bins * 4 if (tracking or adaptive) else 0
    smooth_adds = bins * 3 if (tracking or adaptive) else 0
    track_mults = bins if tracking else 0
    track_adds = bins * 2 if tracking else 0
    snr_logs = 1 if adaptive else 0
    alpha_ops = 8 if adaptive else 0

    return OpsPerFrame(
        n_fft=n_fft,
        hop=hop,
        window_mults=n_fft,                          # analysis window (N)
        fft_mults=butterflies * BUTTERFLY_MULTS,
        fft_adds=butterflies * BUTTERFLY_ADDS,
        magnitude_mults=bins * 2,                    # re^2 + im^2
        magnitude_sqrts=bins,                        # one sqrt per bin
        noise_update_mults=bins * 2 if noise_est else 0,   # recursive avg
        noise_update_adds=bins * 2 if noise_est else 0,
        subtract_mults=bins * 2 if subtract else 0,  # alpha*N/M then gain
        subtract_divs=bins if subtract else 0,       # N[k]/M[k]
        subtract_adds=bins * 2 if subtract else 0,   # 1 - x, clamp vs floor
        reconstruct_mults=bins * 2,                  # gain * (re, im)
        ola_adds=n_fft,                              # overlap-add accumulate
        ola_norm_mults=hop,                          # coverage normalization
        smooth_mults=smooth_mults,
        smooth_adds=smooth_adds,
        track_mults=track_mults,
        track_adds=track_adds,
        snr_logs=snr_logs,
        alpha_ops=alpha_ops,
    )


def np_log2(x: int) -> int:
    """Exact integer log2 (import-local helper; avoids numpy dependency here)."""
    n = 0
    while x > 1:
        x >>= 1
        n += 1
    return n


@dataclass
class PowerEstimate:
    """Labeled result of the power model (never emitted as a measurement)."""

    power_mw: float
    cycles_per_frame: float
    frames_per_second: float
    effective_mhz: float
    crosscheck_uW_per_MHz_range: tuple[float, float] = field(
        default_factory=lambda: _CV32E40P_UW_PER_MHZ_RANGE
    )
    source: str = "cite"
    caveat: str = (
        "MODEL ESTIMATE, not measured on silicon. Computes only the DSP core "
        "compute energy at the cited peak-efficiency operating point; excludes "
        "memory (instruction/data SRAM), leakage, clock tree, I/O, and the "
        "rest of the SoC. See docs/research/power-model.md."
    )
    reference: str = _CV32E40P_MOPS_PER_MW_REF


def estimate_power_mw(n_fft: int, hop: int, fs: int, *,
                      cycle_table: CycleTable | None = None,
                      noise_est: bool = False,
                      subtract: bool = True,
                      tracking: bool = False,
                      adaptive: bool = False) -> PowerEstimate:
    """Estimate per-frame cycles and the resulting core power draw.

    power_mW = cycles_per_frame * frames_per_second / (MOPS_per_mW * 1e6)

    using the cited peak efficiency (193 MOPS/mW). ops ~= cycles at ~1
    cycle/op IPC assumption (RV32IMC in-order); documented, not measured.

    Cross-check view: effective_mhz = cycles_per_frame * frames_per_second /
    1e6, then dynamic power = effective_mhz * uW/MHz(65nm) on both ends of the
    published range.
    """
    manifest = ops_per_frame(n_fft, hop, noise_est=noise_est, subtract=subtract,
                             tracking=tracking, adaptive=adaptive)
    ct = cycle_table or CycleTable()

    cycles = (
        manifest.window_mults * ct.mul
        + manifest.fft_mults * ct.mul
        + manifest.fft_adds * ct.add
        + manifest.magnitude_mults * ct.mul
        + manifest.magnitude_sqrts * ct.sqrt
        + manifest.noise_update_mults * ct.mul
        + manifest.noise_update_adds * ct.add
        + manifest.subtract_mults * ct.mul
        + manifest.subtract_divs * ct.div
        + manifest.subtract_adds * ct.add
        + manifest.reconstruct_mults * ct.mul
        + manifest.ola_adds * ct.add
        + manifest.ola_norm_mults * ct.mul
        + manifest.smooth_mults * ct.mul
        + manifest.smooth_adds * ct.add
        + manifest.track_mults * ct.mul
        + manifest.track_adds * ct.add
        + manifest.snr_logs * ct.log2    # fixed-point log2 (35 cycles, assumed)
        + manifest.alpha_ops * ct.add
    )

    frames_per_second = fs / hop
    cycles_per_second = cycles * frames_per_second
    power_mw = cycles_per_second / (CV32E40P_MOPS_PER_MW * 1e6)
    return PowerEstimate(
        power_mw=power_mw,
        cycles_per_frame=float(cycles),
        frames_per_second=float(frames_per_second),
        effective_mhz=cycles_per_second / 1e6,
    )