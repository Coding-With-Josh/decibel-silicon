# Core selection: CV32E40P (rv32imc, no FPU)

## Requirements driving the choice

Derived from the DSP chain and product framing (hearing aid / low-power
assistive audio):

| constraint | value | source |
|---|---|---|
| DSP energy target | ~20 µW model (DSP core compute only) | `power-model.md` |
| algorithmic latency budget | ≤ 10 ms preferred / 20 ms ceiling | `latency-budget.md` |
| arithmetic | fixed-point Q15 (no FPU) | smaller core, lower energy |
| ISA | RISC-V, open architecture | licensing + toolchain ecosystem |
| load | 250 frames/s × ~15.3 kcycles/frame ≈ 3.8 MHz effective | `power-model.md` |

A 4-bit-per-cycle class microcontroller at a few MHz comfortably covers the
3.8 MHz effective load; the binding constraint is *energy*, not throughput.

## Selected core

**OpenHW CV32E40P** (PULP RI5CY lineage), configured RV32IMC, no FPU.

- Smallest practical general-purpose core that still runs C with the M
  extension (the port uses 32×32→32 multiply, integer divide, and shift;
  no division loops in software).
- 4-stage pipeline, single issue — predictable worst-case timing for the
  frame-period budget.
- Baseline IMC only **for this port**. The core's native Xpulp DSP
  extensions (SIMD MAC, post-increment addressing) are **deferred**; the
  port is built to run on plain rv32imc so the correctness/fidelity result
  does not depend on extensions we do not verify here (documented future
  work: butterfly SIMD versions using `p.sravi`/`p.add` pipelining).

## Evidence in this repo (and what is NOT claimed)

- `firmware/` cross-compiles with
  `riscv64-elf-gcc -march=rv32imc -mabi=ilp32 -Os -ffreestanding` and links
  bare-metal with `-nostdlib -Wl,-e,_start` — no libc, no shims.
  Verified in CI-style `make cross`.
- The port is not simulated on an rv32 target (no simulator/FPGA in this
  repo); "runs on CV32E40P" here means *compile + link verified against the
  ISA* and *host golden-verified for arithmetic fidelity* (see
  `noise-reduction.md`). On-target cycle/energy is modeled, not measured.

## References

1. Gautschi, M., et al. (2017). Near-threshold RISC-V core with DSP
   extensions for scalable IoT endpoint devices. *IEEE Trans. VLSI Systems*,
   25(10), 2700–2713. doi:10.1109/TVLSI.2017.2654506 · arXiv:1608.08376
   *(the CV32E40P / RI5CY paper; source of the µW/MHz energy figures used by
   the power model)*
2. Kim, G., et al. (2019). A 220 µW, 0.023 mm², 1.08 V fully-integrated DSP
   for hearing-aid applications. *IET Circuits, Devices & Systems*, 13(5),
   717–722. doi:10.1049/iet-cds.2018.5374 *(industry-scale hearing-aid DSP
   power reference — our model estimates far below this, but the comparison
   is model-vs-chip and must be labeled as such)*
3. Sokolova, A., et al. (2022). A review of low-power machine learning
   processors for edge applications. *IEEE Access*, 10, 54301–54312.
   doi:10.1109/ACCESS.2022.3176368 *(context for low-cost RISC-V endpoint
   processing; not a direct power source)*

## Alternatives considered and rejected

- **CV32E40S**: same family, smaller (2-stage); rejected because the port
  wants M-extension `div` and the model budget assumes a 4-stage core's
  energy profile (documented, not benchmarked).
- **Ibex (RV32IMC)**: viable alternative; not chosen because the PULP
  lineage keeps the Xpulp SIMD future-work path open.
- **FPU cores (CV32E40P with F, RISC-V F-extension cores)**: rejected —
  the whole chain is deliberately Q15 fixed-point; an FPU would be idle
  silicon and leakage for no benefit.
- **Dedicated HW accelerator for the FFT**: rejected for this scope — the
  port must first be proven on a general-purpose core; an accelerator is
  documented future work if the cycle budget tightens.