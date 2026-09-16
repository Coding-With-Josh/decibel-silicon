# Latency budget: 4.00 ms measured algorithmic delay

## Result (measured, reproducible)

`dsp/benchmark.py` probes the streaming chain with an impulse and reports
the structural delay: **4.00 ms** (64 samples at 16 kHz). Every one of 635
frames passes the 20 ms ceiling; the preferred 10 ms budget is met with
6 ms of headroom. Host compute per frame is ~0.03–0.06 ms worst-case (dev
machine; host-only, not a target figure) — two orders of magnitude below
the 4 ms frame period, so the chain is compute-bound only by the target
clock (see `power-model.md`: 3.82 MHz effective).

## Why it is 4.00 ms and not 8 ms (window-center convention)

A newcomer's first guess for an N=128 (8 ms) frame with hop=64 is an 8 ms
blocking delay. That is wrong for this WOLA layout, and the probe proves it:

- The streaming buffer holds the newest N samples **centered on the frame
  boundary**: at frame k, samples `[(k−1)·hop, (k+1)·hop)`. The synthesis
  block covers the *newest* hop samples, so an impulse at stream position s
  first appears in the output at `out[s] == x[s−hop]` — a constant
  **hop-sample** delay, interior and at the boundary alike
  (verified: impulse probe measures exactly 64 samples).
- The 8 ms figure would only apply if the frame boundary sat at the *start*
  of the analysis buffer (then the newest sample enters N/2 samples late).

**Convention:** "algorithmic delay in this design = hop samples (window
center), placeholder for the device driver/ADC block size budget." The
benchmark's quality metrics delay-compensate by exactly these 64 samples.

## Frame timing

| quantity | value |
|---|---|
| sample rate | 16 kHz |
| frame period | 4.00 ms (64 samples) |
| algorithmic delay | 4.00 ms (= 1 frame period) |
| preferred / ceiling budget | 10 / 20 ms |
| headroom vs ceiling | 16 ms |

## State-machine warm-up (not user-perceivable latency)

- Frame 0: cold start — buffer filling, emits silence (not a partial artifact).
- Frames 1..8: NOISE_EST — the 8-frame noise estimate accumulates; output is
  **passthrough** (gain 1.0), so the leader is delivered unprocessed, never
  "warm-up corrupted".
- Frame 9 onward: ACTIVE — subtraction. The leader-plus-warm-up interval is
  excluded from the quality metrics; it is *budget* (a real deployment plays
  the noise-only leader before speech begins or gates output).

## Tracking

- `dsp/src/pipeline.py`: buffer layout, impulse probe, LatencyReport,
  per-frame ceiling verdicts — the measurement is code, not prose.
- `dsp/benchmark.py`: prints the measured delay and frame pass verdicts.
- `dsp/tests/test_pipeline.py`: impulse-probe and delay-correctness tests.
- Delay-compensation constant used by the quality metrics lives in
  `benchmark.py` and is asserted against the probe.