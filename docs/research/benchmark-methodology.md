# Benchmark methodology

**Goal:** reproducible Blueprint numbers — power (model), latency (measured),
intelligibility (proxy) — with every label honest about what was measured vs
modeled vs proxied.

Run: `dsp/.venv/bin/python dsp/benchmark.py` (default synthetic signal;
`--json out.json` for machine-readable output; `--clean/--noise` wavs to
swap in a real corpus — results then carry a different label, not the
synthetic one).

## Signal synthesis (synthetic mode)

- **Speech-like source**: seeded formant-synthesized stream (2.5 s).
- **Noise source**: low-passed amplitude-modulated babble (the *harder*,
  non-stationary case for a static noise estimate).
- **Mix**: `mix_at_snr(clean, noise, snr_db)` — additive mix at the target
  SNR.
- **Noise-only leader**: the first `noise_frames` frames are the noise
  estimate window, so `build_test_stream` prepends `noise_frames+1` blocks
  of **pure noise scaled to the noise component's RMS** (not the mixture
  RMS — using mixture RMS over-estimates noise by √2 and silently causes
  over-subtraction; this is checked by a test). The leader is excluded from
  quality metrics and the FFT side of the chain evaluates it only as an
  estimate window.

## Metrics

| metric | definition | label |
|---|---|---|
| STOI | pystoi 0.4.1 (Taal et al. 2011, doi:10.1109/TASL.2011.2114881) on synthetic formant speech | *measured proxy* — NOT real-speech intelligibility |
| segSNR | segmental SNR, 32 ms segments, delay-compensated | *measured proxy* |
| algorithmic delay | impulse probe through `pipeline.py` | *measured* |
| power | op-count manifest × cycle costs × cited µW/MHz | *model* (see `power-model.md`) |

**Delay compensation:** quality metrics compare `out[s]` against
`x[s−hop]` (structural delay = 64 samples measured by probe). Without this,
the comparison would charge the chain for the frame delay it is budgeted
for. Latency is reported separately against the 10 ms preferred / 20 ms
ceiling budgets.

## Measured sweep (synthetic formant speech + AM babble)

Config: fs=16 kHz, N=128, hop=64, floor=0.01, noise_frames=8, 9-block
noise leader. Quality region excludes leader + warm-up.

| SNR | α | STOI in → out | ΔSTOI | segSNR in → out | ΔsegSNR |
|---|---|---|---|---|---|
| −5 dB | 1.0 | 0.171 → 0.109 | −0.062 | 0.9 → −0.9 dB | −1.8 dB |
| −5 dB | 2.0 | 0.171 → 0.080 | −0.091 | 0.9 → −1.0 dB | −1.9 dB |
| 0 dB | 1.0 | 0.230 → 0.196 | −0.034 | 5.7 → 3.2 dB | −2.4 dB |
| 0 dB | 2.0 | 0.230 → 0.147 | −0.083 | 5.7 → 1.6 dB | −4.0 dB |
| 5 dB | 1.0 | 0.315 → 0.298 | −0.017 | 10.6 → 7.5 dB | −3.0 dB |
| 5 dB | 2.0 | 0.315 → 0.275 | −0.039 | 10.6 → 5.3 dB | −5.2 dB |
| 10 dB | 1.0 | 0.425 → 0.416 | −0.009 | 15.2 → 12.0 dB | −3.2 dB |
| 10 dB | 2.0 | 0.425 → 0.402 | −0.023 | 15.2 → 9.5 dB | −5.7 dB |
| 15 dB | 1.0 | 0.547 → 0.545 | −0.002 | 19.7 → 16.6 dB | −3.1 dB |
| 15 dB | 2.0 | 0.547 → 0.541 | −0.006 | 19.7 → 14.0 dB | −5.7 dB |
| 20 dB | 1.0 | 0.668 → 0.668 | −0.000 | 23.9 → 21.3 dB | −2.6 dB |
| 20 dB | 2.0 | 0.668 → 0.666 | −0.002 | 23.9 → 18.5 dB | −5.3 dB |

Reading: the static-estimate + full-band baseline preserves intelligibility
proxy only where noise is already mild; it trades segSNR everywhere on this
signal. This is the *expected* baseline limitation, reproduced and
quantified — not a bug (chain correctness is independently verified by WOLA
identity and the tone+white-noise probe, which *improves* segSNR by ~12 dB).
See `noise-reduction.md` for the interpretation and the documented
future-work replacements.

## Reproducibility + honesty rules

- Seed fixed; signal synthesis deterministic; rerunning the default yields
  identical numbers (verified).
- Every printed figure carries a label: **model** / **measured** /
  **measured proxy**; caveats are printed with the number, not in a
  footnote-only appendix.
- A real-corpus run (`--clean/--noise` wavs) replaces the labels and must
  not be quoted as the synthetic result.
- The power estimate is labeled *model* and cross-checked, never presented
  as silicon.

# Revision 2 sweep (adaptive defaults)

**Status:** the revision-1 table above is the honest *motivating* finding; the
revision-2 table below is the current algorithm's. Do not mix the two when
quoting.

Config: fs=16 kHz, N=128, hop=64, floor=0.01, noise_frames=8. v2 defaults:
`noise_tracking=False`, `adaptive_alpha=True` (tent map: alpha=2, ref 10 dB,
slope 0.1, min 1, max 3), `vad_threshold_db=3.0`, `vad_hangover_frames=2`.
Same signal, leader, delays, and labels as the v1 table.

| SNR | STOI in → out | ΔSTOI | segSNR in → out | ΔsegSNR |
|---|---|---|---|---|
| −5 dB | 0.171 → 0.111 | −0.059 | 0.9 → −0.4 dB | −1.2 dB |
| 0 dB | 0.230 → 0.174 | −0.055 | 5.7 → 2.5 dB | −3.2 dB |
| 5 dB | 0.315 → 0.285 | −0.030 | 10.6 → 5.6 dB | −4.9 dB |
| 10 dB | 0.425 → 0.412 | −0.013 | 15.2 → 10.7 dB | −4.6 dB |
| 15 dB | 0.547 → 0.547 | −0.000 | 19.7 → 16.4 dB | −3.3 dB |
| 20 dB | 0.668 → 0.668 | −0.000 | 23.9 → 21.3 dB | −2.6 dB |

Reproduce: `dsp/.venv/bin/python dsp/benchmark.py --sweep snr-db:-5,0,5,10,15,20
--sweep-json out.json` (default adaptive config).

## Variation cells (measured, same signal)

| config | 0 dB ΔSTOI | 0 dB ΔsegSNR | 20 dB ΔsegSNR |
|---|---|---|---|
| v1 static α=2 (baseline) | −0.083 | −4.0 dB | −5.3 dB |
| v2 default (adaptive α only) | **−0.055** | **−3.2 dB** | **−2.6 dB** |
| v2 + tracking μ=0.15 (downward-only gate) | −0.054 | −3.1 dB | −2.5 dB |
| v2 + tracking μ=0.05 | −0.055 | −3.2 dB | −2.6 dB |

## Reading the revision-2 numbers

1. The tent-shaped SNR→α map is the win: 0 dB STOI loss shrinks 34%
   (−0.083 → −0.055), segSNR −4.0 → −3.2 dB, and 20 dB now matches the
   best v1 cell (α=1) that the static default could not reach — the map
   relaxes α at both SNR extremes (see `noise-reduction.md`).
2. VAD-gated tracking (even with the downward-only gate that fixed the
   measured warm-up cascade) is within run-to-run noise of tracking-off —
   it is kept OFF by default and documented as a deployment option for
   genuinely non-stationary noise.
3. **The 0 dB STOI gap is PARTIALLY closed, not closed.** Every low-SNR
   cell still degrades the proxies; verdict labels in docs never claim
   otherwise.

## Cost (model, not a measurement)

Default adaptive config: 16,566 cycles/frame (15,288 + 1,278 adaptive) →
21.46 µW at the cited operating point (+8.4%). With tracking enabled: 16,956
→ 21.96 µW. Latency unchanged: 4.00 ms measured, 635/635 frames under the
20 ms ceiling.