# Decibel Silicon — firmware: fixed-point noise reduction port

Fixed-point (Q15) port of the validated floating-point baseline in
`dsp/src/spectral_subtraction.py` (Boll 1979 magnitude-averaging noise estimate,
Berouti et al. 1979 over-subtraction, periodic Hann^2 WOLA filter bank).

Target: OpenHW **CV32E40P** (RV32IMC, no FPU — see `docs/research/core-selection.md`).
Build tested with `riscv64-elf-gcc -march=rv32imc -mabi=ilp32 -Os -ffreestanding`.
Xpulp/PULP DSP extensions (SIMD, post-increment addressing) are **deferred** to a
documented future work item; everything here runs on baseline IMC.

## Layout

```
src/fixed_math.h         Q15 helpers: saturating add/mul, Q15 division, u32 sqrt
src/fft.h  src/fft.c     radix-2 DIT complex FFT, per-stage halving (total 1/N);
                         inverse via conjugate trick (no 1/N)
src/spectral_subtract.h  nss_t state + constants (N=128, H=64, NOISE_FRAMES=8,
.c                       alpha=2.0, floor=0.01) and the streaming frame API
src/tables.h            GENERATED: twiddles, Hann^2 window, coverage reciprocal
test/golden.h           GENERATED: deterministic 48-frame stream + reference output
test/host_test.c        golden comparison (host) — the fidelity gate
test/fw_linktest.c      freestanding _start link check (cross)
scripts/gen_tables.py   regenerates tables + golden from the validated Python
Makefile                gen-tables | host-test | cross | all
```

## Build & verify

```sh
make all        # gen-tables -> host-test (golden gate) -> cross (rv32imc link)
make host-test  # runs the golden comparison on the host
make cross      # rv32imc objects + bare-metal link check (no libc)
```

## Accuracy vs the reference (measured, reproducible)

Generated from the same Q15-quantized input; the fixed-point chain is compared
sample-for-sample against the validated float reference:

| metric | value | meaning |
|---|---|---|
| max abs diff | 545 Q15 | 1.66 % of full scale (single worst sample, frame 7) |
| rms diff | 118.8 Q15 | 0.36 % of full scale |
| gate | max diff ≤ 1024 Q15 | fail-closed on gross mismatch (scale/state/gain bugs produce 30–100× larger diffs) |

This is the expected accuracy class of a scaled 128-point radix-2 Q15 FFT + Q15
WOLA; it is recorded here because it is a *measurement*, not an assumption.

## Scale conventions (each tied to the reference + golden)

- **Q15** everywhere (int16), tables are literal integers (no host math at
  runtime; twiddles/window/coverage come from `tables.h`).
- **Forward FFT**: per-stage halving → net 1/N scaling, matching the
  reference's `rfft()` band scale. The Berouti gain is a *ratio* `N[k]/|X[k]|`,
  so the gain value is scale-invariant — the golden comparison checks the
  actual output, not an intermediate.
- **Inverse FFT**: `conj(fft(conj(X))) × N` — net unscaled inverse, matching
  `np.fft.irfft()`.
- **Noise estimate**: sum-then-shift (`>>3`) at the 8th frame; the `>>3` is
  only valid because `NSS_NOISE_FRAMES == 8` (WARNING comment at the site).
- **WOLA normalization** `out = acc·inv_cov` is computed as
  `acc + (acc·(inv_cov−32768)) >> 15` to keep every intermediate inside
  int32 (naive `acc·inv_cov` overflows INT32_MAX in the coverage-trough
  region — overflow guard, commented at the site).

## Implemented / simulated / missing

| item | status | notes |
|---|---|---|
| Q15 FFT (forward + inverse) | **implemented**, host-tested | bit-reverse permute, per-stage halving, conj inverse |
| Berouti gain + spectral floor | **implemented**, host-tested | Q15 division, int32 α product (α=2 bound-checked) |
| Streaming state machine (COLD→NOISE_EST→ACTIVE) | **implemented**, host-tested | mirrors reference exactly, incl. flip-frame behavior |
| WOLA synthesis + coverage normalization | **implemented**, host-tested | overflow-safe split multiply |
| rv32imc freestanding build + link | **implemented**, verified | `-nostdlib -Wl,-e,_start`, no libc symbols |
| On-target execution (RV32 execution of instructions) | **simulated** | cross-compile + link only; no rv32 simulator/FPGA in this repo |
| Cycle/energy estimate | **modeled in dsp/src/power_model.py** | not measured on silicon; see docs/research/power-model.md |
| Xpulp SIMD versions of FFT butterflies | **missing (deferred)** | documented future work; port is portable IMC baseline |

## Firmware contract (fail-closed)

- `nss_process_frame` emits exactly HOP samples for HOP consumed; the cold-start
  frames emit **silence**, never partial artifacts.
- NOISE_EST frames are **passthrough**; no subtraction artifact is ever
  attributed to processing during warm-up.
- Any undefined arithmetic (division by non-positive denominator) yields **0**
  (deny), never a garbage gain.
- The chain is **not re-entrant** — `nss_process_frame` must not be preempted
  mid-frame; one instance per device (documented at the site).