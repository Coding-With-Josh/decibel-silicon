# Learned GRU band-gain model — real held-out comparison (v2, post gain_hook fix)

**Headline (honest, N=1 held-out slice: DEMAND PCAFETER@5 dB, 15 s, fs=16 kHz; relay
auditing rule means the learned path must GENUINELY engage or the run aborts):**

| metric                | classical              | learned (GRU)          |
|-----------------------|------------------------|------------------------|
| STOI in→out           | 0.8482 → 0.8167 ℹ | 0.8482 → **0.8546** |
| segSNR in→out (dB)    | 0.3160 → 1.2732    | 0.3160 → **2.9311** |
| on-host power est.    | 21.5 µW                | 74.5 µW (×3.5)         |

Learned **holds/improves STOI on held-out speech** (post-fix), where classical
degrades it — and adds ~+1.7 dB segSNR. **Honest about the flip side**: the GRU
decision costs ~3.5× the classical per-block alpha decision on this host
(nothing is quantized/embedded yet; power is a float32 on-host estimate, not a
target/firmware number).

## What CHANGED to make this real (the bug, owned)
Pre-fix `benchmark.py` installed the learned gains **below** the first
`NoiseReductionPipeline(...)` construction. On the user-wavs (`--clean/--noise`)
path the reference to `gain_hook` inside that construction was evaluated before
the function-local had been assigned → `UnboundLocalError`; the fix **hoists the
learned block above the first pipeline construction** (`gain_hook = None` bound
at line 318, first pipeline at 333 — verified via `grep` on the staged file)
and wires `gain_hook` into the pipeline the same way the synthetic path is wired.
Result: `--gain-model` now drives genuinely learned gains that OBSERVABLY diverge
from classical (probe: 3750/3750 active frames differ; learned gain range
0.060→0.785), instead of a silent classical fallback that kept the file
byte-identical while a "learned" row was printed — THAT was the pre-fix
hook-not-engaging bug (benchmark.py constructed the first pipeline BEFORE
binding gain_hook, so --gain-model silently fell back to the classical path).
Fixed + verified: with gain_hook bound (line 316) before the first pipeline,
the learned hook genuinely engages (3750/3750 active frames, learned gains
0.060→0.785, outputs DIVERGE — see the v2 held-out numbers above).

## How the learned decision is made (one paragraph)
`learned/model.py` GRU (float32, hidden=32, ~8.6k params, IRM-trained) maps the
pipeline's own band-RMS magnitudes through the SAME window/FFT/noise-estate the
classical algorithm maintained; the GRU emits per-band gains that
`_apply_gain_hook` clamps to [0,1] and substitutes for the classical
alpha*noise/mag reduction *only on ACTIVE frames*. Everything downstream
(window, WOLA, spectral-floor) is the untouched classical code — so the ONLY
difference the benchmark can show between the two runs is the gain decision
itself publishers. That is the comparison the table above reports.

## NOT claimed (stays true)
- Not quantized to int8/float16; not on-target; not in the firmware path.
- STOI/segSNR are normal-hearing proxy metrics — no HAAQI/STOI-IP claim.
- N=1 held-out slice; a multi-category held-out sweep is the next step.
- Power is an on-host float32 model estimate via `estimte_learned_power_mw`;
  on-target GRU cost would be measured by the repo's WOLA power meter, not this
  float32 arithmetic.

## Reproduce
```
.venv/bin/python benchmark.py --clean corpus/librispeech_clean_3sent.wav \
    --noise corpus/demand_pcafeter_noise_15s.wav --fs 16000 --snr-db 5 \
    --gain-model learned/runs/gru_v1.npz --json corpus/_cache/holdout_pcafeter_learned_v2.json
```
Classical comparison drops `--gain-model`. Both rc=0 today (learned v2
STOI_out 0.8546, segSNR_out 2.9311 dB, power 74.5 µW).
