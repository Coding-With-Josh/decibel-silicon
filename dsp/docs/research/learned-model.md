# Learned GRU gain model — honest status (do not quote wins)

**Headline (held-out PCAFETER, 5 dB): learned output == classical output.**
| metric        | classical | learned | verdict              |
|---------------|-----------|---------|----------------------|
| STOI  in→out  | 0.848→0.817 | 0.848→0.817 | identical — hook did NOT change audio |
| segSNR in→out | 0.316→1.273 dB | 0.316→1.273 dB | identical |
| est. power    | 21.459 µW | 21.459 µW | identical |
Only µs-scale per-frame latency jitter differs between the two runs — that is
system noise, not a model effect.

**The numbers are REAL from today's run logs + JSONs** (`corpus/_cache/learned_dkitchen.json`,
`corpus/_cache/holdout_pcafeter_{classical,learned}.json`), and the honest reading is:

## What is DONE and verified
- `learned/` package: float32 GRU (hidden=32, bands=24, 8664 idx-params incl. bias),
  numpy/pyTorch-gate parity asserted in `tests/test_learned_model.py` (12 tests).
- Training: real run, epochs 1..20, `learned/runs/gru_v1.npz` (loads+validates),
  best val MSE recorded in that run's log. Schema: float32 arrays, shape-validated.
- Wiring: `benchmark.py --gain-model` loads + schema-shape-sanity checks and ABORTS
  (exit 2) on any mismatch BEFORE the pipeline runs — no silent classical fallback
  when `--gain-model` is explicitly requested. `gain_hook` is bound before any
  pipeline construction (this was a real UnboundLocalError fixed in this pass).
- Classical path left fully intact: classical benchmark rc=0, tests still pass.

## What is HONESTLY NOT achieved
- **The hook did not change the output.** held-out PCAFETER + DKITCHEN runs under
  `--gain-model` produce byte-identical headline metrics to the classical runs.
  A prototype that doesn't change the audio yet is NOT an improvement to report.
- No per-category win table exists. Do NOT quote any "learned beats classical"
  number — none was measured.
- Not quantized, not on target, not in firmware path.

## Why (working hypothesis, for the next pass)
The learned GRU's gains are clamped/mapped through the same band-gain → alpha/bin
path that classical uses; with 24 bands folded onto 65 bins and sigmoid outputs
close to the classical decision, the *effective* decision is unchanged. The feature
needs the classical-vs-learned *decision hook* to actually branch on the model
(A/B the gains), which the current `NoiseReductionPipeline` does not yet do —
it applies the hook's gains but there is no learned-specific *decision* to flip.

## Next steps (from here, honest)
1. Instrument `GainModelHook` to log min/max actual gain deltas per run (prove
   engagement, not guess).
2. Make the pipel line apply the learned gain VERBATIM (no fold/clamp back to the
   classical decision) so the model can actually diverge; re-measure held-out.
3. Only then write a learned-vs-classical table.

This file exists so future reads know exactly what was / wasn't established.

---
## v2 appendix — held-out (PCAFETER, 5 dB) measured today
Run via `benchmark.py` (classical rc=0, learned rc=0), JSONs in
`corpus/_cache/holdout_pcafeter_{classical,learned}.json`:

| path | classical | learned |
|---|---|---|
| STOI in→out | 0.8482→0.8167 | 0.8482→0.8167 |
| segSNR in→out | 0.3160→1.2732 dB | 0.3160→1.2732 dB |
| est. power | 21.459 µW | 21.459 µW |

Identical to 4dp across the board; only `compute_ms_host_only` per-frame jitter
differs (µs-scale, system noise). Conclusion: **the learned gain hook does not
yet engage on held-out audio — output is byte-identical to classical.** This is
the true, current state; no "learned beats classical" number is claimed anywhere
in this repo.
