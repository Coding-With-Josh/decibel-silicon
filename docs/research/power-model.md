# Power model: DSP-core compute energy estimate

**Status:** MODEL, not a measurement. Reproducible via
`dsp/.venv/bin/python dsp/benchmark.py` (default run prints the estimate).

## How the number is produced (no free parameters)

1. **Operation-count manifest** — every op in the per-frame chain is counted
   from the exact structure of `dsp/src/spectral_subtraction.py` (the code
   is the manifest; the fixed-point port `firmware/` mirrors the same
   structure so the count tracks the device implementation):

   | step | operations per frame (N=128, H=64) |
   |---|---|
   | window (N mults) | 128 mul |
   | radix-2 FFT, 7 stages × 64 butterflies | 7·64 = 448 butterflies × (4 mul + 8 add + index work) |
   | magnitude (65 bins: 2 mul + 1 add + 1 sqrt) | 130 mul + 65 add + 65 sqrt |
   | noise accumulation (65 bins, NOISE_EST only) | 65 add |
   | Berouti gain (ACTIVE; 65 bins: 1 div + 1 mul + compare) | 65 div + 65 mul |
   | gain apply (128 bins × 2 mults) | 256 mul |
   | IFFT (same as FFT) | 448 butterflies |
   | synthesis window (128 mul) | 128 mul |
   | OLA (128 add) + coverage normal (64 div-equiv) | 128 add + 64 mult+shift |
   | shift bookkeeping (loops) | ~256 add/moves |
   | **v2 adaptive meters** (ACTIVE only; adaptive_alpha or noise_tracking on) | smooth: 65×4 mul + 65×3 add; energies: same counters; |
   | **v2 tracking** (only when noise_tracking on) | 65 mul + 130 add on downward-gated frames |
   | **v2 SNR→alpha map** | 1 fixed-point log2 (~35 cycles, ASSUMED) + ~8 ops |

   Cycle costs (rv32imc, no FPU): add/move=1, mul=4, div=10, sqrt via
   non-restoring binary (~20 cycles for 32-bit), log2 via exponent+linear
   interpolation (~35 cycles, assumed until cycle-accurate profiling). Total
   **15 288 cycles/frame** for the v1 static path (asserted by
   `power_model.py`); the v2 default adds **1 278** (adaptive meters + tent
   map) → **16 566**; enabling tracking adds 390 more → **16 956**. The
   ported loops in `firmware/` stay within the same op classes (the device
   is still the v1 static port this pass; see `noise-reduction.md`).

2. **Energy** — E = cycles/frame × frames/s × µW/MHz.

   - frames/s = 250 (fs/hop).
   - effective clock = 15 288 × 250 = **3.822 MHz** (v1) /
     16 566 × 250 = **4.142 MHz** (v2 default) / 16 956 × 250 = **4.239 MHz**
     (v2 + tracking).
   - µW/MHz: Gautschi et al. report 193 MOPS/mW at 40 MHz / 1 mW (28 nm
     FD-SOI near-threshold, CV32E40P-class core) → 5.18 µW/MHz nominal.
   - **v1: 19.80 µW. v2 default: 21.46 µW. v2 + tracking: 21.96 µW** =
     effective-MHz × 5.18.

3. **Cross-check** — the 65 nm range from the same paper (12.26–33.8
   µW/MHz at 1.08 V): 4.142 × [12.26, 33.8] = **[50.8, 140.0] µW** (v2
   default; v1 was [46.9, 129.2] µW). The
   28 nm number is the headline; the 65 nm range bounds the process
   sensitivity. Reference: Gautschi et al. 2017,
   doi:10.1109/TVLSI.2017.2654506 (values quoted in `firmware/README` and
   `core-selection.md`).

## What is excluded (labeled, not hidden)

- Instruction/data SRAM access energy, leakage, clock tree, I/O pads, the
  rest of the SoC, and the ADC/DAC path.
- Real benchmark on silicon (no tape-out / no FPGA power measurement here).
- Xpulp SIMD variants (deferred) would cut the butterfly cycle count; not
  reflected.

**Claim:** "the DSP core compute for this chain is modeled at ~20 µW
(v1 static) / ~21.5 µW (v2 adaptive) at 28 nm, / ~50–140 µW at 65 nm". It is
not a chip measurement, and the docs never present it as one.

## Tracking

- `dsp/src/power_model.py`: manifest + cycle costs (source of truth).
- `dsp/benchmark.py`: prints the estimate with source=citation label.
- `firmware/README.md`: notes the port maps to the same op classes.
- `core-selection.md`: why a few-MHz rv32imc core suffices.
- Kim et al. 2019 (IET CDS 13(5):717–722) gives a 220 µW fabricated
  hearing-aid DSP as an industry scale point — a model-vs-chip comparison
  that must be labeled, never merged into the headline number.