"""Noise reduction: adaptive spectral subtraction (Boll 1979, Berouti 1979).

This module is the DSP-side algorithm for Decibel Silicon's hearing-aid chain.
Revision 2 (this file) replaces the static-estimate baseline with an adaptive
one; the v1 behavior is preserved behind config flags and documented as the
motivating finding (see docs/research/noise-reduction.md).

Algorithm per frame (WOLA filter-bank, N=128 @ 16 kHz, hop=64):
  1. Window the latest n_fft samples (periodic Hann^2).
  2. rFFT -> complex spectrum and magnitude |X|.
  3. Noise estimate:
       - NOISE_EST state (first `noise_frames`): average magnitude of the
         noise-only leader (unchanged v1 behavior; quiet-period fitting).
- ACTIVE state, IF noise_tracking is enabled: a voice-activity detector
          (per-frame total-energy ratio + hangover) gates a recursive noise
          update  N[k] += mu * (mag[k] - N[k])  on frames that are BOTH
          speech-inactive AND meaningfully quieter than the current estimate
          (ratio_db < TRACK_MIN_RATIO_DB). The downward-only gate is the
          measured fix for "amplitude-modulated noise outruns the static
          estimate": the estimate descends toward the noise's quieter moments
          and can never chase speech upward (a not-speech-only gate let the
          estimate follow a steady tone - 251 updates, alpha_eff collapse).
          It is the pragmatic, fixed-point-cheap relative of Martin's
          minimum-statistics tracking (rejected here as v1.5: sliding-window
          minima cost W x bins of RAM and a per-frame search on a core whose
          SRAM budget is tight; see docs/research/noise-reduction.md).
4. Berouti over-subtraction with spectral floor, with an SNR-adaptive
      over-subtraction factor (IF adaptive_alpha):
          alpha_eff = clamp(alpha - slope * |snr_est_db - snr_ref_db|,
                            alpha_min, alpha_max)
      snr_est_db is a smoothed global signal-to-noise ratio estimated from the
      smoothed magnitude energy vs the current noise estimate energy. The map
      is a TENT centred at snr_ref_db: alpha_eff equals alpha at mid SNR and
      relaxes toward alpha_min at BOTH extremes. That shape is what the v1
      benchmark sweep demanded empirically: at low SNR a high alpha zeroes
      bins where |X| ~ 2N (STOI collapse; measured 0 dB STOI 0.230->0.147),
      and at high SNR over-subtraction creates musical-noise artifacts (the
      canonical Berouti SNR->alpha map also decreases alpha as SNR rises).
      Identity-test modes are bypassed explicitly: if alpha <= 0 or floor >= 1
      the gain is forced to 1.0 and no adaptive mapping runs.
  5. Apply the real gain to the complex spectrum (no atan2); 6. IFFT;
  7. synthesis window + overlap-add with coverage normalization (WOLA
     identity holds exactly for gain=1: verified by tests).

Provenance / labeling rules (Phase 1 mediation):
  - No measured power/latency claim originates here; the operation-count
    manifest used by power_model.py tracks this exact structure (adaptive
    ops included) - see docs/research/power-model.md.
  - Behavioral claims about the adaptive approach come from benchmark.py
    runs and this test suite, never from literature read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Default parameters (canonical; see docs/research/latency-budget.md)
# --------------------------------------------------------------------------
DEFAULT_FS = 16000          # Hz; hearing-aid sample rate (Kim et al. 2019)
DEFAULT_N_FFT = 128         # 8 ms frame at 16 kHz
DEFAULT_HOP = 64            # 50% overlap
DEFAULT_ALPHA = 2.0         # base over-subtraction factor (Berouti)
DEFAULT_FLOOR = 0.01        # spectral floor beta
DEFAULT_NOISE_FRAMES = 8    # frames averaged into the leader estimate

# --- adaptive-revision parameters ------------------------------------------
# VAD-gated noise tracking (v2). Default OFF by measurement, not default-on:
# on the benchmark stream, enabling tracking at 0 dB SNR makes the STOI loss
# WORSE (-0.104..-0.120 vs -0.083 static) because the global-energy VAD cannot
# separate speech from noise at low SNR and the estimate chases the mixture.
# The mechanism is kept, tested, and documented: it only pays off when the
# noise is non-stationary (AM troughs) AND the SNR supports the VAD gate.
DEFAULT_NOISE_TRACKING = False    # recursive noise update on inactive frames
DEFAULT_NOISE_UPDATE_MU = 0.15    # EMA coefficient for the noise update [0,1)
DEFAULT_VAD_THRESHOLD_DB = 3.0    # frame-energy ratio (dB) => "speech active"
DEFAULT_VAD_HANGOVER_FRAMES = 2   # frames held active after a VAD trigger
# SNR-adaptive over-subtraction (v2). Default ON and the headline fix:
# measured 0 dB STOI loss -0.083 (static) -> -0.055 at default alpha map.
DEFAULT_ADAPTIVE_ALPHA = True
DEFAULT_ALPHA_SNR_REF_DB = 10.0   # snr_est at which alpha_eff == alpha
DEFAULT_ALPHA_SNR_SLOPE = 0.1     # tent half-slope: alpha_eff per dB away from ref
DEFAULT_ALPHA_MIN = 1.0           # alpha_eff floor (protects speech at 0 dB)
DEFAULT_ALPHA_MAX = 3.0           # alpha_eff ceiling (validation dome)

# Internal smoothing constants (documented, deliberately not config):
VAD_MAG_SMOOTHING = 0.2    # per-bin magnitude smoothing for the VAD/SNR meter
SNR_EST_SMOOTHING = 0.3    # frame-to-frame smoothing of the SNR estimate

# Tracking only fires when the current frame reads MEANINGFULLY QUIETER than
# the current estimate (ratio_db < TRACK_MIN_RATIO_DB). Reasoning (measured):
# with a mere "not-speech" gate (ratio < 3 dB) the tracker chases the mixture
# during the smoothed meter's warm-up - the estimate rises toward speech, the
# ratio collapses below the gate, and every subsequent speech frame keeps
# updating (251 updates on a steady tone; alpha_eff collapsed to 1.0). A
# strictly-downward gate terminates that cascade: estimates can only descend
# toward quieter noise (the AM-trough fix) and can never follow speech.
TRACK_MIN_RATIO_DB = -0.5

STATE_COLD_START = "cold_start"
STATE_NOISE_EST = "noise_est"
STATE_ACTIVE = "active"


@dataclass
class SpectralSubtractionConfig:
    """Validated configuration for the noise reducer.

    Validation happens at construction (fail-closed: reject, don't coerce).
    """

    fs: int = DEFAULT_FS
    n_fft: int = DEFAULT_N_FFT
    hop: int = DEFAULT_HOP
    alpha: float = DEFAULT_ALPHA
    floor: float = DEFAULT_FLOOR
    noise_frames: int = DEFAULT_NOISE_FRAMES
    # v2 adaptive parameters (see module docstring).
    noise_tracking: bool = DEFAULT_NOISE_TRACKING
    noise_update_mu: float = DEFAULT_NOISE_UPDATE_MU
    vad_threshold_db: float = DEFAULT_VAD_THRESHOLD_DB
    vad_hangover_frames: int = DEFAULT_VAD_HANGOVER_FRAMES
    adaptive_alpha: bool = DEFAULT_ADAPTIVE_ALPHA
    alpha_snr_ref_db: float = DEFAULT_ALPHA_SNR_REF_DB
    alpha_snr_slope: float = DEFAULT_ALPHA_SNR_SLOPE
    alpha_min: float = DEFAULT_ALPHA_MIN
    alpha_max: float = DEFAULT_ALPHA_MAX

    def __post_init__(self) -> None:
        if not isinstance(self.fs, int) or self.fs <= 0:
            raise ValueError(f"fs must be a positive integer, got {self.fs!r}")
        if not isinstance(self.n_fft, int) or self.n_fft < 8:
            raise ValueError(f"n_fft must be an integer >= 8, got {self.n_fft!r}")
        if self.n_fft & (self.n_fft - 1):
            raise ValueError(f"n_fft must be a power of two, got {self.n_fft}")
        if not isinstance(self.hop, int) or self.hop <= 0:
            raise ValueError(f"hop must be a positive integer, got {self.hop!r}")
        if self.hop > self.n_fft:
            raise ValueError(f"hop ({self.hop}) must be <= n_fft ({self.n_fft})")
        # The overlap-add bookkeeping in this implementation assumes the hop
        # divides the FFT size (standard for WOLA filter banks at 2x overlap).
        if self.n_fft % self.hop != 0:
            raise ValueError(
                f"n_fft must be a multiple of hop for this WOLA implementation "
                f"({self.n_fft} % {self.hop} != 0)"
            )
        # alpha == 0.0 is an explicit "no subtraction" mode (gain == 1) used
        # by WOLA-identity tests; the subtraction domain is otherwise (0, 10].
        if not 0.0 <= self.alpha <= 10.0:
            raise ValueError(f"alpha must be in [0, 10], got {self.alpha!r}")
        # floor == 1.0 is an explicit "gain pinned to 1" mode used by WOLA-
        # identity tests; the clamping domain is otherwise [0, 1).
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {self.floor!r}")
        if not isinstance(self.noise_frames, int) or self.noise_frames < 1:
            raise ValueError(
                f"noise_frames must be a positive integer, got {self.noise_frames!r}"
            )
        # --- v2 validation (all fail-closed) -------------------------------
        if not isinstance(self.noise_tracking, bool):
            raise ValueError(
                f"noise_tracking must be a bool, got {self.noise_tracking!r}"
            )
        # mu == 0 disables tracking; mu >= 1 would let one frame's magnitude
        # overwrite the estimate (never allowed - fail-closed).
        if not 0.0 <= self.noise_update_mu < 1.0:
            raise ValueError(
                f"noise_update_mu must be in [0, 1), got {self.noise_update_mu!r}"
            )
        if not self.vad_threshold_db > 0.0:
            raise ValueError(
                f"vad_threshold_db must be > 0, got {self.vad_threshold_db!r}"
            )
        if (not isinstance(self.vad_hangover_frames, int)
                or self.vad_hangover_frames < 0):
            raise ValueError(
                "vad_hangover_frames must be an integer >= 0, "
                f"got {self.vad_hangover_frames!r}"
            )
        if not isinstance(self.adaptive_alpha, bool):
            raise ValueError(
                f"adaptive_alpha must be a bool, got {self.adaptive_alpha!r}"
            )
        if not np.isfinite(self.alpha_snr_ref_db):
            raise ValueError(
                f"alpha_snr_ref_db must be finite, got {self.alpha_snr_ref_db!r}"
            )
        # Bound the slope: a runaway signed slope could drive alpha_eff out of
        # the subtraction domain even WITH clamping (it would just sit at a
        # bad clamp forever). Keep it a sane per-SNR rate.
        if not -0.5 <= self.alpha_snr_slope <= 0.5:
            raise ValueError(
                f"alpha_snr_slope must be in [-0.5, 0.5], "
                f"got {self.alpha_snr_slope!r}"
            )
        if not 0.0 <= self.alpha_min <= self.alpha_max <= 10.0:
            raise ValueError(
                f"need 0 <= alpha_min <= alpha_max <= 10, got "
                f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max}"
            )


class SpectralSubtraction:
    """Streaming adaptive spectral-subtraction noise reducer.

    Caller contract (sequential streaming, Phase 3 - no re-entrancy):
      - process_frame() consumes exactly `hop` new samples and returns exactly
        `hop` output samples. State mutates only at frame boundaries.
      - Use reset() to start a new stream. Replaying identical input after
        reset() produces identical output.
      - Any exception leaves the object in a defined state: the current frame
        is NOT committed (compute into locals, commit at the end).
      - v2 state is deterministic: the noise-tracking/VAD/SNR meters are pure
        functions of the input stream (reset() -> identical replay).
    """

    def __init__(self, config: SpectralSubtractionConfig | None = None) -> None:
        self.config = config or SpectralSubtractionConfig()
        cfg = self.config
        # Periodic Hann analysis window (w[0] small but nonzero):
        # sin^2(pi * n / N) for n in 0..N-1.
        n = np.arange(cfg.n_fft)
        self.window = np.sin(np.pi * n / cfg.n_fft) ** 2
        # WOLA coverage normalization: sum over overlapping frames of w[j]^2.
        # gain=1 then reconstructs the input exactly (identity test depends on
        # this).
        self._inv_coverage = self._compute_inv_coverage()
        # Last applied over-subtraction factor (introspection/tests; updated
        # on every ACTIVE frame).
        self.last_alpha_eff = float(cfg.alpha)
        self.reset()

    # -- public API ---------------------------------------------------------

    def reset(self) -> None:
        cfg = self.config
        self._state = STATE_COLD_START
        self._buffer = np.zeros(cfg.n_fft, dtype=np.float64)
        self._ola_acc = np.zeros(cfg.n_fft, dtype=np.float64)
        self._noise_mag = np.zeros(cfg.n_fft // 2 + 1, dtype=np.float64)
        self._frames_in_noise_est = 0
        self._frame_index = 0
        self._total_consumed = 0
        self._total_emitted = 0
        self._first_full_frame = False
        # v2 meter state (all deterministic).
        self._sm_mag = np.zeros(cfg.n_fft // 2 + 1, dtype=np.float64)
        self._sm_running = False
        self._energy_sig = 0.0
        self._energy_noise = 0.0
        self._snr_est_db: float | None = None
        self._vad_hangover = 0
        self._noise_update_count = 0
        self.last_alpha_eff = float(cfg.alpha)

    @property
    def state(self) -> str:
        return self._state

    @property
    def noise_magnitude(self) -> np.ndarray:
        """Copy of the current noise magnitude estimate (bins N/2+1).

        A read-only snapshot for tests/instrumentation; the firmware port
        mirrors this array exactly, so tests can assert tracking behavior
        against the same quantity the port will hold.
        """
        return self._noise_mag.copy()

    @property
    def noise_update_count(self) -> int:
        """Number of ACTIVE frames on which noise tracking updated the estimate."""
        return self._noise_update_count

    def process_frame(self, new_samples: np.ndarray) -> np.ndarray:
        """Consume `hop` new samples; return `hop` processed output samples."""
        cfg = self.config
        new_samples = np.asarray(new_samples, dtype=np.float64)
        if new_samples.size != cfg.hop:
            raise ValueError(
                f"process_frame expects exactly hop={cfg.hop} samples, "
                f"got {new_samples.size}"
            )
        # Perimeter validation (Phase 1): reject non-finite samples outright.
        if not np.isfinite(new_samples).all():
            raise ValueError("process_frame received non-finite samples (NaN/Inf); refusing")

        # Shift buffer left by hop, append new samples (ring of last n_fft).
        self._buffer[: cfg.n_fft - cfg.hop] = self._buffer[cfg.hop :]
        self._buffer[cfg.n_fft - cfg.hop :] = new_samples
        self._total_consumed += cfg.hop

        if self._state == STATE_COLD_START and self._total_consumed < cfg.n_fft:
            # Not a full frame yet: emit silence rather than a partial artifact.
            out = np.zeros(cfg.hop, dtype=np.float64)
            self._total_emitted += cfg.hop
            return out

        if self._state == STATE_COLD_START:
            self._state = STATE_NOISE_EST

        # Compute into locals; commit state only after a fully valid frame.
        windowed = self._buffer * self.window
        spec = np.fft.rfft(windowed, n=cfg.n_fft)
        mag = np.abs(spec)
        n_bins = mag.size

        if self._state == STATE_NOISE_EST:
            # Accumulate magnitude spectrum into the noise estimate (v1 path
            # unchanged: leader average, sum-then-divide).
            self._noise_mag += mag
            self._frames_in_noise_est += 1
            if self._frames_in_noise_est >= cfg.noise_frames:
                self._noise_mag /= cfg.noise_frames
                self._state = STATE_ACTIVE

        if self._state == STATE_ACTIVE:
            # --- v2 adaptive meters (ACTIVE only) ---------------------------
            # The VAD/SNR machinery consumes the FINAL noise estimate, so it
            # cannot run before the flip-frame division; starting it here also
            # keeps NOISE_EST cost identical to v1.
            need_meters = cfg.noise_tracking or (
                cfg.adaptive_alpha and cfg.alpha > 0.0 and cfg.floor < 1.0
            )
            if need_meters:
                # Smoothed magnitude (for stable energy/SNR meters).
                if not self._sm_running:
                    self._sm_mag = mag.copy()
                    self._sm_running = True
                else:
                    self._sm_mag = (
                        (1.0 - VAD_MAG_SMOOTHING) * self._sm_mag
                        + VAD_MAG_SMOOTHING * mag
                    )
                # eps-guarded energies => ratio floor is finite (-120 dB);
                # NaN can never reach the VAD/gain path (fail-closed).
                self._energy_sig = float(np.sum(self._sm_mag ** 2))
                self._energy_noise = float(np.sum(self._noise_mag ** 2))
                ratio_db = 10.0 * np.log10(
                    (self._energy_sig + 1e-12) / (self._energy_noise + 1e-12)
                )

                # Voice-activity decision with hangover: once VAD triggers,
                # it stays active for hangover frames so trailing speech
                # energy cannot leak a speech frame into the noise estimate.
                if self._vad_hangover > 0:
                    self._vad_hangover -= 1
                    vad_active = True
                else:
                    vad_active = bool(ratio_db >= cfg.vad_threshold_db)
                    if vad_active:
                        self._vad_hangover = cfg.vad_hangover_frames

                # Noise tracking on frames that are BOTH speech-inactive (VAD + hangover)
                # AND meaningfully quieter than the current estimate. The
                # downward-only gate is what keeps the tracker from chasing
                # speech: once the estimate is at the mixture level the frame
                # ratio is ~0 dB and updates stop (no cascade - see
                # TRACK_MIN_RATIO_DB). Deterministic: mu fixed.
                if (cfg.noise_tracking and not vad_active
                        and ratio_db < TRACK_MIN_RATIO_DB):
                    self._noise_mag += cfg.noise_update_mu * (mag - self._noise_mag)
                    self._noise_update_count += 1

                # SNR estimate (smoothed) -> adaptive alpha_eff.
                if (cfg.adaptive_alpha and cfg.alpha > 0.0 and cfg.floor < 1.0):
                    if self._snr_est_db is None:
                        self._snr_est_db = float(ratio_db)
                    else:
                        self._snr_est_db = (
                            (1.0 - SNR_EST_SMOOTHING) * self._snr_est_db
                            + SNR_EST_SMOOTHING * float(ratio_db)
                        )
                    # Tent-shaped SNR->alpha map (see module docstring):
                    # alpha_eff == alpha at snr_ref_db, relaxes toward
                    # alpha_min on BOTH sides (protects buried speech at low
                    # SNR and clean signal at high SNR).
                    alpha_eff = float(np.clip(
                        cfg.alpha
                        - cfg.alpha_snr_slope
                        * abs(self._snr_est_db - cfg.alpha_snr_ref_db),
                        cfg.alpha_min,
                        cfg.alpha_max,
                    ))
                else:
                    alpha_eff = float(cfg.alpha)
                self.last_alpha_eff = alpha_eff
            else:
                # v1 static path (identical to baseline behavior).
                alpha_eff = float(cfg.alpha)
                self.last_alpha_eff = alpha_eff

            # Berouti over-subtraction with spectral floor, alpha_eff.
            # gain[k] = max(1 - alpha_eff * N[k] / max(|X[k]|, eps), floor)
            denom = np.maximum(mag, 1e-12)
            gain = np.maximum(1.0 - alpha_eff * self._noise_mag / denom, cfg.floor)
            enhanced = gain * spec
        else:
            # NOISE_EST warm-up: passthrough. No subtraction artifact is
            # attributed to processing during warm-up (Phase 2).
            enhanced = spec
            self.last_alpha_eff = float(cfg.alpha)

        time_signal = np.fft.irfft(enhanced, n=cfg.n_fft)

        # Synthesis window + overlap-add contribution at OLA positions 0..N-1
        # relative to this frame's output block base.
        contribution = self.window * time_signal
        self._ola_acc += contribution

        out = self._ola_acc[: cfg.hop] * self._inv_coverage[: cfg.hop]

        # Shift OLA accumulator left by hop; carry tail, zero the vacuum.
        self._ola_acc[: cfg.n_fft - cfg.hop] = self._ola_acc[cfg.hop :]
        self._ola_acc[cfg.n_fft - cfg.hop :] = 0.0

        self._frame_index += 1
        self._total_emitted += cfg.hop
        return out

    # -- internals ----------------------------------------------------------

    def _compute_inv_coverage(self) -> np.ndarray:
        """Per-position inverse window-power coverage, period H.

        output[p] = acc[p] / cov[p] where cov[p] = sum_m w[p - m*H]^2 over the
        frames m whose synthesis window covers position p. With gain=1 the
        chain reconstructs the input exactly when acc[p] = w[a]^2 * x[p] summed
        over the same frames (verified by the identity test).
        """
        cfg = self.config
        w2 = self.window**2
        cov = np.zeros(cfg.hop, dtype=np.float64)
        for m in range(-(cfg.n_fft // cfg.hop) + 1, 1):
            # Window m covers positions [m*H, m*H + N); contribution to the
            # coverage at position p in [0, H) is w2[p - m*H] when 0 <= p-m*H < N.
            start = m * cfg.hop
            for p in range(cfg.hop):
                j = p - start
                if 0 <= j < cfg.n_fft:
                    cov[p] += w2[j]
        cov = np.maximum(cov, 1e-12)
        return 1.0 / cov