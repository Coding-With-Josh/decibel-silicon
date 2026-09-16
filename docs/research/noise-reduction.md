# Noise reduction: baseline choice (spectral subtraction)

**Status:** implemented (floating-point prototype + fixed-point port), benchmarked,
honestly characterized. This page records *why* the baseline is what it is, and
what it measurably does and does not do.

## Algorithm (steps 1–6 = v1 static framework; v2 adaptive revision below)

Per-frame WOLA filter-bank (N=128 @ 16 kHz, hop=64, 50% overlap, periodic
Hann² analysis/synthesis windows):

1. Window the latest N samples.
2. rFFT → complex spectrum and magnitude |X[k]|.
3. Noise magnitude estimate N[k]: the average of the magnitude spectra of the
   first `noise_frames` frames (Boll's magnitude-averaging proposal
   [1]). During this NOISE_EST state the gain is pinned to 1.0 (passthrough)
   so warm-up never emits subtraction artifacts.
4. Berouti over-subtraction with a spectral floor [2]:

   gain[k] = max(1 − alpha·N[k]/|X[k]|, floor),   alpha = 2.0, floor = 0.01

5. Apply the real gain to the complex spectrum (no atan2/phase re-synthesis —
   the fixed-point-friendly form used by `firmware/`).
6. IFFT, synthesis window, overlap-add with per-position coverage
   normalization so gain=1 reconstructs the input exactly (WOLA identity,
   verified: clean-stream through-ratio is 1.0000 to float precision).

## What the baseline measurably does

Measured on the synthetic benchmark (formant speech + low-passed
amplitude-modulated babble; methodology and full sweep table in
`benchmark-methodology.md`):

- **Delay:** 4.00 ms algorithmic delay (window-center convention, measured by
  impulse probe) — well inside the 10/20 ms preferred/ceiling budget.
- **Power model:** 19.80 µW DSP-core compute estimate at the cited
  peak-efficiency operating point (see `power-model.md`; model, not silicon).
- **Noise-only suppression:** −8.7 dB on the noise-only probe.
- **Controlled tone+white-noise probe:** ~+12 dB segSNR improvement.

## Honest baseline limitation (measured, not a guess)

On the synthetic formant speech + amplitude-modulated babble, the baseline
**degrades** the quality proxies at every tested SNR (−5…20 dB) and α (1, 2):

| condition | dSTOI | dsegSNR |
|---|---|---|
| 0 dB, α=2 (default) | −0.083 | −4.0 dB |
| 0 dB, α=1 | −0.034 | −2.4 dB |
| 10 dB, α=1 | −0.009 | −3.2 dB |
| 20 dB, α=1 | −0.000 | −2.6 dB |

Root causes (known spectral-subtraction failure modes, reproduced here):

1. **Static 8-frame noise estimate**: the estimate is the mean over the
   leader; amplitude-modulated babble bursts are louder than the mean, so
   during bursts the subtraction is under-strength, and between bursts the
   residual noise is musical.
2. **Over-subtraction of speech**: full-band α·N attenutes inter-harmonic
   speech energy (harmonics are sparse in the formant signal); this is why
   segSNR drops even at 20 dB where STOI is preserved.
3. **Musical noise floor**: the floor (0.01) leaves audible tonality.

The STOI preservation at high SNR proves the chain (WOLA identity, scaling,
delay) is correct; the degradation at low SNR is the algorithm's known
limitation, not an implementation bug.

**Documented future work** (not implemented): noise-tracking (minima
statistics, MCRA), DNN enhancement. Each must be justified against the
power/latency budget before replacing the baseline — the fixed-point port
and power model track the current structure deliberately.
*(The static-estimate + SNR-constant-α items listed here as future work in
revision 1 were implemented as revision 2 — see below.)*

# Revision 2: measurement-driven adaptive fixes

**Status:** implemented (float prototype; firmware port deferred to a later
pass), benchmarked, honestly characterized. This section records what the v1
baseline did wrong, which of the two candidate fixes measurably worked, and
what the numbers now say.

## v1 finding that motivated the revision

The static 8-frame noise estimate + full-band α=2 subtraction **degraded
STOI at low SNR** on the synthetic benchmark: 0 dB STOI 0.230→0.147
(Δ−0.083), segSNR −4.0 dB. Two candidate fixes were specified (task), both
implemented, both measured:

## Candidate A — VAD-gated noise tracking → implemented, default OFF

`N[k] += μ·(mag[k] − N[k])` on frames the VAD marks as speech-inactive. The
task explicitly requested "adaptive noise tracking"; Martin's
minimum-statistics [3] is the literature's answer but was **rejected for this
core**: a W-frame sliding minimum costs W × bins of RAM plus a per-frame
search — the SRAM budget on the chosen core is tight (see
`core-selection.md`). A recursive EMA is fixed-point-cheap, which is why it
was chosen.

**The first gate semantics failed under measurement.** Gating on "not
speech" (frame energy ratio < 3 dB) let tracking fire on the smoothed
meter's warm-up frames: the estimate rose toward the speech+noise mixture,
the ratio collapsed below 3 dB, and every later speech frame kept updating —
**251 of 160 speech frames** on a steady tone, with alpha_eff collapsed to
1.0. The gate was changed to **downward-only**: track only frames that are
both speech-inactive AND **meaningfully quieter than the current estimate**
(ratio_db < −0.5 dB). The estimate can then only descend toward quieter
noise (the AM-trough fix) and can never chase speech upward; the cascade is
gone (measured: 0 speech-frame updates on the same tone, alpha_eff ≈ 1.9 at
10 dB instead of 1.0).

**Why the default is still OFF:** even with the downward-only gate the
synthetic-sweep cells with tracking enabled are within noise of the
tracking-off cells (0 dB: ΔSTOI −0.054 vs −0.055, ΔsegSNR −3.07 vs −3.21),
so the extra state buys nothing on this signal and adds a VAD-threshold
tuning surface and 390 cycles/frame. The mechanism is kept, tested, and
documented for deployment on genuinely non-stationary noise where the v1
static estimate's AM-outrun failure is the dominant term; the right call
here is measurement over theory.

## Candidate B — SNR-adaptive over-subtraction factor → implemented,
**default ON, headline fix**

`alpha_eff = clamp(alpha − slope·|snr_est_db − ref_db|, alpha_min, alpha_max)`
with a **tent-shaped** map: alpha_eff equals `alpha` at mid SNR and relaxes
toward `alpha_min` at BOTH extremes. The shape is dictated by the v1 sweep,
not by taste:

- at low SNR a high α zeroes bins where |X| ≈ 2N (0 dB STOI collapse);
- at high SNR over-subtraction creates musical noise (v1's best 20 dB cell
  was α=1, not the α=2 default) — the canonical Berouti SNR→α guidance [2]
  also *decreases* α as SNR rises.

**Mapping-shape caveat (read before quoting):** the tent shape — α_eff
*peaking* at mid SNR and relaxing toward `alpha_min` at BOTH extremes — is
**this project's empirical finding on this specific synthetic test signal,
not a literature-standard Berouti curve.** The classical Berouti guidance is
monotonically *decreasing* in SNR (more aggressive subtraction at low SNR,
less at high); this project's low-SNR relaxation is a defensible departure
(over-subtraction destroying buried speech at very low SNR is a documented
failure mode — see Measured result below), but nobody should read this doc
as claiming the tent is textbook. If a future signal/corpus shows the low-SNR
relaxation is wrong, the map is a config (slope/min/max/ref), not a rewrite.

snr_est_db is the eps-guarded ratio of smoothed-mixture energy to the
noise-estimate energy (a fixed-point-cheap global measure; the meter floor
is 0 dB, so cells below ~−10 dB real SNR read ≈0 dB and land on alpha_min —
fail-closed, no NaN path).

Identity-test modes (α ≤ 0 or floor ≥ 1) bypass the mapping outright — the
WOLA identity is preserved exactly (tests).

## Measured result (synthetic formant speech + AM babble; full table in
`benchmark-methodology.md`)

| condition | v1 static (α=2) | v2 default | verdict |
|---|---|---|---|
| 0 dB ΔSTOI | −0.083 | **−0.055** | gap *partially* closed (34% of loss recovered) |
| 0 dB ΔsegSNR | −4.0 dB | **−3.2 dB** | 20% recovered |
| 5 dB ΔSTOI | −0.039 | **−0.030** | improved |
| 15 dB ΔSTOI | −0.006 @α2 | **−0.000** | flat (v1-best) |
| 20 dB ΔSTOI | −0.002 @α2 | **−0.000** | flat (v1 α=1 best) |
| 20 dB ΔsegSNR | −5.3 dB @α2 | **−2.6 dB** | matches v1 α=1 best |

**Honest verdict:** the revision partially closes the 0 dB gap (STOI loss
−0.083 → −0.055; segSNR −4.0 → −3.2). The **residual high-SNR loss is closed
by revision 2.1 below** (20 dB segSNR −2.6 dB → −0.0 dB). The low-SNR STOI
dip **remains negative** — the algorithm still damages intelligibility where
the noise is loud. Closing it further (Wiener/MMSE gains, per-band
modulation, or a DNN stage) is beyond this pass and is documented as future
work, not claimed.

## Cost of the revision (model, not silicon)

Adaptive meters + tent map: +1,278 cycles/frame (15,288 → 16,566) → 21.46 µW
at the cited operating point (+8.4% over the v1 19.80 µW). Tracking (when
enabled) adds 390 more → 21.96 µW. See `power-model.md`. Latency unchanged:
4.00 ms measured (the meters run inside the ACTIVE frame; no added buffering).

## Revision 2.1 — high-SNR bypass (default OFF; `high_snr_bypass_db`)

The tent map relaxed α at high SNR but subtraction still ran (20 dB ΔsegSNR
was −2.6 dB — over-subtraction of inter-harmonic speech energy). The fix is
not to tune α again but to **stop processing clean-ish frames entirely**: once
the smoothed SNR meter (the same `snr_est_db` the alpha map uses — no second
estimator) strictly exceeds `high_snr_bypass_db`, the frame passes through
EXACTLY — gain=1 through the verified WOLA identity — and the subtraction
arithmetic and alpha map never run.

- Decision is per-frame and idempotent; fail-closed: if the meter cannot
  decide (`snr_est_db` is None) the bypass is DENIED → subtraction.
- Validation: `None` (off) or finite `> 0` dB; `≤ 0` would classify
  near-silence as clean and is rejected. Operating range: ≥ 10 dB.
- Measured on the benchmark (threshold 15 dB): **0 dB → 0/627 frames
  bypassed (inert), 10 dB → 247/627 (+0.004 ΔSTOI), 15 dB → 620/627
  (ΔSTOI +0.000, ΔsegSNR −0.0 dB), 20 dB → 622/627 (−0.0 dB)**. The
  high-SNR gap is now FULLY closed at the cells where the meter reads ≥ 15 dB;
  low-SNR cells are unchanged (partial closure stands).
- Power (model): a bypassed frame skips subtraction + alpha map → 15,258
  cycles → 19.76 µW per frame; a stream that mixes bypassed and active frames
  draws between the two — the benchmark prints worst-case, bypass-case, and
  the measured bypassed/ACTIVE counts.

## References

1. Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral
   subtraction. *IEEE Trans. Acoustics, Speech, and Signal Processing*,
   27(2), 113–120. doi:10.1109/TASSP.1979.1163209
2. Berouti, M., Schwartz, R., & Makhoul, J. (1979). Enhancement of speech
   corrupted by acoustic noise. *ICASSP '79*, 4, 208–211.
   doi:10.1109/ICASSP.1979.1170788 *(DOI resolved 2026-09-15: IEEE Xplore
   record for the ICASSP '79 paper)*
3. Martin, R. (2001). Noise power spectral density estimation based on
   optimal smoothing and minimum statistics. *IEEE Trans. Speech and Audio
   Processing*, 9(5), 504–512. doi:10.1109/89.928915 *(verified resolves via
   Crossref API 2026-09-16; title/venue match. Correction: an earlier draft
   cited 10.1109/89.928615 — a transposed digit — and wrongly concluded the
   old IEEE record predates the Crossref deposit; a 404 and a typo produce
   identical symptoms, so the fix is re-search + digit-by-digit comparison,
   not "record missing".)*
4. Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An
   algorithm for intelligibility prediction of time-frequency weighted noisy
   speech. *IEEE Trans. Audio, Speech, and Language Processing*, 19(7),
   2125–2136. doi:10.1109/TASL.2011.2114881 *(metric used by the benchmark)*

## Fixed-point port fidelity (firmware/)

**Note (revision 2):** the firmware port still implements the STATIC
revision-1 algorithm (steps 1–6 above, alpha constant). Re-porting the
adaptive revision onto the device is a separate pass — the device table
generator (`firmware/gen_tables.py`) currently imports the dsp module, so
regenerating tables after this revision would silently change the firmware's
behavior; that regeneration is deliberately NOT run in this pass.

The Q15 port re-implements steps 1–6 exactly (see `firmware/README.md`).
Measured against the float reference on a deterministic 48-frame stream:

| metric | value |
|---|---|
| max abs diff | 545 Q15 = 1.66 % full scale (worst frame 7) |
| rms diff | 118.8 Q15 = 0.36 % full scale |
| gate | max diff ≤ 1024 Q15 (fail-closed on gross mismatch) |

Regenerate the golden and re-run with `make all` in `firmware/`.