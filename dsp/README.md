# Decibel Silicon — DSP research pipeline (`dsp/`)

Floating-point prototype of the noise-reduction chain that will run on the
low-power hearing-aid chip. The fixed-point port for the RISC-V target lives
in `firmware/` (same algorithm, Q15 arithmetic, no FPU).

## What's here

| Path | Purpose |
|---|---|
| `src/spectral_subtraction.py` | Baseline noise reducer (Boll 1979 + Berouti over-subtraction / spectral floor), streaming WOLA filter bank. **Floating-point prototype — needs fixed-point port** (provided in `firmware/`). |
| `src/power_model.py` | Compute-cycle and power estimate (cycles/frame → mW) using the cited CV32E40P peak efficiency (193 MOPS/mW, Gautschi et al. 2017). **Model estimate, not a measurement.** |
| `src/pipeline.py` | Streaming harness: measures algorithmic delay via an impulse probe, reports PASS/FAIL per frame against the 10–20 ms Blueprint ceiling. |
| `benchmark.py` | Emits the three Blueprint numbers: power estimate, latency vs ceiling, intelligibility proxy (STOI + segSNR) — each labeled with provenance. |
| `tests/` | pytest suite for all of the above. |

## Setup (this machine)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy pytest pystoi
```

## Run

```bash
.venv/bin/python -m pytest tests/            # tests
.venv/bin/python benchmark.py                # benchmark (synthetic signal)
.venv/bin/python benchmark.py --json out.json --duration-s 4
```

## Honesty rules (enforced by the repo's research policy)

- Every benchmark number carries a `source` tag: `measurement`, `cite (model)`,
  or `proxy`. Nothing is presented as measured unless it was.
- The power draw is a compute-core-only estimate at the cited peak-efficiency
  operating point; it excludes memory, leakage, I/O. See
  `docs/research/power-model.md`.
- The latency number is *measured* (impulse probe) but reflects this
  implementation's WOLA alignment; the reference figure in the literature is
  a design target, not a substitute for the measurement.
- STOI/segSNR are proxies on a synthetic speech-like signal by default;
  supply `--clean`/`--noise` wavs to run on real audio, and label the corpus
  before quoting as speech intelligibility.
- Host compute times (`compute_ms`) are **dev-machine-only** and are never
  presented as target-silicon execution time.