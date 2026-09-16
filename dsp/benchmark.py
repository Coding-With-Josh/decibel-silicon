#!/usr/bin/env python3
"""Benchmark harness: the three Blueprint numbers with provenance labels.

Emits, for the configured pipeline, exactly the three numbers the Blueprint
goal asks for - and labels each one with how it was obtained:

  1. power draw estimate [mW]         source="cite (model)"  - NOT a measurement
  2. latency [ms] vs the 10-20 ms     source="measured (impulse probe)" +
     ceiling (PASS/FAIL per frame)    per-frame host-compute rows (dev-only)
  3. speech-intelligibility proxy     source="measured proxy (pystoi STOI /
     (STOI + segmental SNR)           segSNR) on a SYNTHETIC speech-like
                                      signal" - proxy until real-corpus tests

Every number carries a provenance tag (Phase 1 mediation): no unlabeled number
leaves this module. The exact configuration used is echoed next to the
numbers so a parameter change can never silently invalidate a quoted figure.

Usage:
    uv run --python .venv/bin/python benchmark.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from src.pipeline import (
    DEFAULT_CEILING_MS,
    DEFAULT_PREFERRED_MS,
    LatencyReport,
    NoiseReductionPipeline,
    run_stream,
)
from src.power_model import estimate_power_mw
from src.spectral_subtraction import (
    DEFAULT_ADAPTIVE_ALPHA,
    DEFAULT_ALPHA,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_SNR_REF_DB,
    DEFAULT_ALPHA_SNR_SLOPE,
    DEFAULT_FLOOR,
    DEFAULT_FS,
    DEFAULT_HIGH_SNR_BYPASS_DB,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_NOISE_FRAMES,
    DEFAULT_NOISE_TRACKING,
    DEFAULT_NOISE_UPDATE_MU,
    DEFAULT_VAD_HANGOVER_FRAMES,
    DEFAULT_VAD_THRESHOLD_DB,
    SpectralSubtractionConfig,
)


# --------------------------------------------------------------------------
# Synthetic test signal (deterministic, no external corpus - Phase 2 rule
# about licensing/network: the default signal is generated, not downloaded,
# and is explicitly labeled synthetic everywhere it flows).
# --------------------------------------------------------------------------

_SPEC_LIKE_CAVEAT = (
    "proxy on a SYNTHETIC speech-like signal (formant-synthesized, seeded); "
    "NOT a real-speech intelligibility number. Replace with a public corpus "
    "run before quoting as speech-intelligibility."
)


def synthesize_speech_like(fs: int, duration_s: float, seed: int = 0) -> np.ndarray:
    """Deterministic speech-like excitation: harmonic source, time-varying
    F0, and vowel-formant (F1/F2) resonators so the STFT carries real
    spectro-temporal modulation (STOI needs envelope patterns, a steady tone
    would degenerate it).
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s + 1)
    t = np.arange(n) / fs

    f0 = 110.0 * (1.0 + 0.12 * np.sin(2 * np.pi * 0.7 * t)
                  + 0.05 * np.sin(2 * np.pi * 0.23 * t))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    src = np.zeros(n)
    for h in range(1, 21):
        amp = rng.uniform(0.35, 1.0) / h  # ~1/h harmonic rolloff
        src += amp * np.sin(h * phase + rng.uniform(0.0, 2 * np.pi))

    # Vowel targets (approx F1, F2 Hz): a / i / u, 0.5 s each, deterministic.
    vowels = np.array([[700.0, 1100.0], [350.0, 2200.0], [300.0, 2600.0]])
    idx = (t // 0.5).astype(int) % len(vowels)
    f1 = vowels[idx, 0]
    f2 = vowels[idx, 1]

    def resonator(x: np.ndarray, theta: float, r: float) -> np.ndarray:
        # y[n] = x[n] + 2 r cos(theta) y[n-1] - r^2 y[n-2]
        b = [1.0]
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        return lfilter(b, a, x)

    out = np.zeros_like(src)
    seg = int(fs * 0.5)
    r = 0.95
    for s in range(0, n, seg):
        part = src[s : s + seg]
        if part.size == 0:
            break
        y = resonator(part, 2 * np.pi * f1[s] / fs, r)
        y = resonator(y, 2 * np.pi * f2[s] / fs, r)
        out[s : s + seg] = y

    peak = np.max(np.abs(out)) or 1.0
    return (out * 0.5 / peak).astype(np.float64)


def synthesize_babble_noise(fs: int, duration_s: float, seed: int = 7) -> np.ndarray:
    """Low-passed + amplitude-modulated noise as a crude babble stand-in."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s + 1)
    white = rng.standard_normal(n)
    # 2nd-order low-pass ~ 800 Hz
    b, a = _lowpass_coeffs(fs, 800.0)
    low = lfilter(b, a, white)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * np.arange(n) / fs
                             + rng.uniform(0, 2 * np.pi))
    return (low * mod).astype(np.float64)


def _lowpass_coeffs(fs: float, cutoff: float):
    """RBJ-style 2nd-order low-pass, returned UN-normalized.

    lfilter() normalizes by a[0] internally; pre-dividing by a0 here would
    double-normalize and can push a pole outside the unit circle (observed:
    |z|=1.67 -> NaN blow-up). Keep a[0] = a0 and let lfilter normalize.
    """
    w = 2 * np.pi * cutoff / fs
    q = 0.707
    alpha = np.sin(w) / (2 * q)
    a0 = 1 + alpha
    b = [(1 - np.cos(w)) / 2, 1 - np.cos(w), (1 - np.cos(w)) / 2]
    a = [a0, -2 * np.cos(w), 1 - alpha]
    return np.array(b), np.array(a)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so that 10*log10(P_clean/P_noise) == snr_db."""
    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2) or 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0)))
    return clean + scale * noise


# --------------------------------------------------------------------------
# Quality metrics (proxies - always labeled as such)
# --------------------------------------------------------------------------


def segmental_snr(clean: np.ndarray, processed: np.ndarray, fs: int,
                  frame_ms: int = 30, hop_ms: int = 10,
                  snr_min: float = -10.0, snr_max: float = 35.0,
                  source_kind: str = "synthetic signal") -> dict:
    """Per-frame SNR over speech-active frames, clipped to [snr_min, snr_max].

    Same frame geometry as the STOI proxy; returns the label + the value so
    callers always surface the proxy/measurement distinction. The label
    carries the source kind so a real-corpus run is never quoted as the
    synthetic result.
    """
    frame = int(fs * frame_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    vals = []
    for start in range(0, min(clean.size, processed.size) - frame, hop):
        c = clean[start : start + frame]
        p = processed[start : start + frame]
        pc = float(np.mean(c**2))
        if pc < 1e-10:  # silence frame: skip (matches speech-active convention)
            continue
        err = float(np.mean((c - p) ** 2)) + 1e-12
        snr = 10.0 * np.log10(pc / err)
        vals.append(float(np.clip(snr, snr_min, snr_max)))
    value = float(np.mean(vals)) if vals else float("nan")
    return {"value": value, "n_frames": len(vals),
            "label": f"measured proxy (segmental SNR, {source_kind})"}


def stoi_proxy(clean: np.ndarray, processed: np.ndarray, fs: int,
               source_kind: str = "synthetic signal") -> dict:
    """STOI (Taal et al. 2011) via pystoi. Degrades WITH LABEL, never crashes.

    Phase 2 fallback rule: if pystoi is unavailable or rejects the signal,
    return value=None and a reason; the caller must surface the absence
    rather than fabricate a number. The label carries the source kind.
    """
    try:
        from pystoi import stoi
    except Exception as exc:  # pragma: no cover - env-dependent
        return {"value": None, "reason": f"pystoi unavailable: {exc}",
                "label": "unavailable"}
    try:
        value = float(stoi(clean, processed, fs, extended=False))
    except Exception as exc:
        return {"value": None, "reason": f"pystoi rejected signal: {exc}",
                "label": "unavailable"}
    return {"value": value, "reason": None,
            "label": f"measured proxy (STOI, pystoi, {source_kind})"}


# --------------------------------------------------------------------------
# Benchmark orchestration
# --------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    config: dict
    power: dict
    latency: LatencyReport
    stoi_input: dict
    stoi_output: dict
    segsnr_input: dict
    segsnr_output: dict
    signal_note: str = _SPEC_LIKE_CAVEAT

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "power_estimate_mw": self.power,
            "latency": {
                "measured_algo_delay_ms": self.latency.measured_algo_delay_ms,
                "ceiling_ms": self.latency.ceiling_ms,
                "preferred_ms": self.latency.preferred_ms,
                "total_frames": self.latency.total_frames,
                "passed_frames": self.latency.passed_frames,
                "overall_pass": self.latency.passed_ceiling,
                "worst_compute_ms_host_only": self.latency.worst_compute_ms,
                "per_frame_verdicts": [
                    {"frame": f.frame_index, "algo_delay_ms": f.algo_delay_ms,
                     "compute_ms_host_only": f.compute_ms,
                     "pass": f.passed_ceiling}
                    for f in self.latency.frames
                ],
            },
            "intelligibility_proxy": {
                "stoi_input": self.stoi_input,
                "stoi_output": self.stoi_output,
                "segsnr_input": self.segsnr_input,
                "segsnr_output": self.segsnr_output,
                "delay_compensation_samples": self.config["hop"],
                "noise_leader_blocks": self.config["noise_leader_blocks"],
                "note": ("quality metrics compare delay-compensated signals "
                         "(structural delay = hop samples measured by probe); "
                         "metrics exclude the noise-only leader and warm-up; "
                         "latency is budgeted separately"),
                "caveat": self.signal_note,
            },
        }


def build_test_stream(cfg: SpectralSubtractionConfig,
                      clean: np.ndarray, noise: np.ndarray, snr_db: float,
                      noise_frames: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Assemble the benchmark stream: a noise-only leader, then speech+noise.

    The baseline's noise estimate is the average magnitude of the FIRST
    frames. For the estimate to be valid (and for the benchmark numbers to
    reflect real use), the stream must start with noise-only audio - exactly
    the quiet-period assumption a hearing-aid fitting makes.

    The leader is the SAME scaled noise used in the speech region (one scale
    computed once), so the estimate matches the speech-region noise statistics
    exactly. Sizing the leader to the mixture RMS instead would over-estimate
    the noise by sqrt(2) at 0 dB and over-subtract.

    Returns (noisy_full, clean_full, lead_len): clean_full has `lead_len`
    leading zeros so both arrays align sample-for-sample.
    """
    lead_blocks = noise_frames + 1  # covers cold-start silence + accumulation
    lead_len = lead_blocks * cfg.hop
    # One scale for the whole stream: noise is attenuated to hit snr_db
    # relative to clean.
    p_clean = float(np.mean(clean**2))
    p_noise = float(np.mean(noise**2)) or 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0)))
    scaled_noise = scale * noise
    speech_region = clean + scaled_noise
    noisy_full = np.concatenate([scaled_noise[:lead_len], speech_region])
    clean_full = np.concatenate([np.zeros(lead_len), clean])
    return noisy_full, clean_full, lead_len


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    fs = args.fs
    cfg = SpectralSubtractionConfig(
        fs=fs, n_fft=args.n_fft, hop=args.hop,
        alpha=args.alpha, floor=args.floor, noise_frames=args.noise_frames,
        noise_tracking=args.noise_tracking,
        noise_update_mu=args.noise_update_mu,
        vad_threshold_db=args.vad_threshold_db,
        vad_hangover_frames=args.vad_hangover_frames,
        adaptive_alpha=args.adaptive_alpha,
        alpha_snr_ref_db=args.alpha_snr_ref_db,
        alpha_snr_slope=args.alpha_snr_slope,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        high_snr_bypass_db=args.high_snr_bypass_db,
    )
    pipeline = NoiseReductionPipeline(
        config=cfg, ceiling_ms=args.ceiling_ms, preferred_ms=args.preferred_ms,
    )

    # ---- input signal -----------------------------------------------------
    if args.clean and args.noise:
        from scipy.io import wavfile
        sig_src = "user wav files"
        fs_clean, clean = wavfile.read(args.clean)
        fs_noise, noise = wavfile.read(args.noise)
        if fs_clean != fs_noise:
            raise ValueError("--clean and --noise must share a sample rate")
        fs = int(fs_clean)
        cfg = SpectralSubtractionConfig(
            fs=fs, n_fft=args.n_fft, hop=args.hop,
            alpha=args.alpha, floor=args.floor, noise_frames=args.noise_frames,
            noise_tracking=args.noise_tracking,
            noise_update_mu=args.noise_update_mu,
            vad_threshold_db=args.vad_threshold_db,
            vad_hangover_frames=args.vad_hangover_frames,
            adaptive_alpha=args.adaptive_alpha,
            alpha_snr_ref_db=args.alpha_snr_ref_db,
            alpha_snr_slope=args.alpha_snr_slope,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            high_snr_bypass_db=args.high_snr_bypass_db,
        )
        pipeline = NoiseReductionPipeline(
            config=cfg, ceiling_ms=args.ceiling_ms, preferred_ms=args.preferred_ms,
        )
        if clean.ndim != 1 or noise.ndim != 1:
            raise ValueError("only mono wav files are supported")
        clean = clean.astype(np.float64) / (np.iinfo(clean.dtype).max + 1.0)
        noise = noise.astype(np.float64) / (np.iinfo(noise.dtype).max + 1.0)
        min_len = min(clean.size, noise.size)
        clean, noise = clean[:min_len], noise[:min_len]
        clean = clean / (np.max(np.abs(clean)) or 1.0)
        if args.corpus_name:
            source_kind = f"real speech (corpus: {args.corpus_name})"
            signal_note = (
                f"real-speech corpus [{args.corpus_name}], "
                "normal-hearing proxy metrics on user-supplied wav files; "
                "NOT impaired-hearing validation (HASPI/HAAQI or listener "
                "testing is the open next step, not claimed here)"
            )
        else:
            source_kind = "real speech (user-supplied wav files)"
            signal_note = (
                "user-supplied wav files (label the corpus before quoting; "
                "normal-hearing proxy metrics only)"
            )
    else:
        sig_src = f"synthetic (seed={args.seed})"
        clean = synthesize_speech_like(fs, args.duration_s, seed=args.seed)
        noise = synthesize_babble_noise(fs, args.duration_s, seed=args.seed + 1)
        source_kind = "synthetic signal"
        signal_note = _SPEC_LIKE_CAVEAT

    noisy = mix_at_snr(clean, noise, args.snr_db)
    # Perimeter check (Phase 1): refuse non-finite samples before they can
    # poison metrics.
    for name, sig in (("clean", clean), ("noisy", noisy)):
        if not np.isfinite(sig).all():
            raise ValueError(f"{name} contains non-finite samples; aborting")

    # Noise-only leader so the initial noise estimate is valid (see
    # build_test_stream docstring).
    noisy_full, clean_full, lead_len = build_test_stream(
        cfg, clean, noise, args.snr_db, cfg.noise_frames
    )
    if not np.isfinite(noisy_full).all():
        raise ValueError("assembled stream contains non-finite samples; aborting")

    # ---- run pipeline -----------------------------------------------------
    report = run_stream(pipeline, noisy_full)
    processed = _reconstruct_output(pipeline, noisy_full)

    # ---- exclude leader + warm-up passthrough from QUALITY metrics --------
    # lead_len >= ACTIVE start (noise_frames frames), so everything from
    # lead_len on is real subtraction. The impulse-probe-measured structural
    # delay (hop samples) is compensated: out[s] == in[s - hop] interior.
    skip = lead_len
    delay_comp = cfg.hop
    c = clean_full[skip:-delay_comp]
    nz = noisy_full[skip:-delay_comp]
    proc = processed[skip + delay_comp :]
    if c.size < fs:  # metrics need meaningful length
        raise ValueError("signal too short after warm-up exclusion; "
                         "increase --duration-s")

    stoi_in = stoi_proxy(c, nz, fs, source_kind=source_kind)
    stoi_out = stoi_proxy(c, proc, fs, source_kind=source_kind)
    seg_in = segmental_snr(c, nz, fs, source_kind=source_kind)
    seg_out = segmental_snr(c, proc, fs, source_kind=source_kind)

    # ---- power model ------------------------------------------------------
    est = estimate_power_mw(cfg.n_fft, cfg.hop, cfg.fs,
                            tracking=cfg.noise_tracking,
                            adaptive=cfg.adaptive_alpha)
    power = {
        "value_mw": est.power_mw,
        "cycles_per_frame": est.cycles_per_frame,
        "frames_per_second": est.frames_per_second,
        "effective_mhz": est.effective_mhz,
        "source": "cite (model)",
        "reference": est.reference,
        "crosscheck_uW_per_MHz_range": list(est.crosscheck_uW_per_MHz_range),
        "caveat": est.caveat,
    }
    if cfg.high_snr_bypass_db is not None:
        # Revision 2.1: report BOTH the all-subtract worst case (headline,
        # unchanged) and the per-frame cost when a frame IS bypassed, plus the
        # actually-bypassed fraction of ACTIVE frames measured in the replayed
        # run - never one disguised number (Phase 2 power-claim honesty).
        est_byp = estimate_power_mw(cfg.n_fft, cfg.hop, cfg.fs,
                                    tracking=cfg.noise_tracking,
                                    adaptive=cfg.adaptive_alpha,
                                    bypass=True)
        power["bypass_value_mw"] = est_byp.power_mw
        power["bypass_cycles_per_frame"] = est_byp.cycles_per_frame
        power["bypassed_frames"] = pipeline.bypass_count
        power["active_frames"] = pipeline.active_frames

    config_dict = {
        "fs": cfg.fs, "n_fft": cfg.n_fft, "hop": cfg.hop,
        "alpha": cfg.alpha, "floor": cfg.floor,
        "noise_frames": cfg.noise_frames,
        "noise_tracking": cfg.noise_tracking,
        "noise_update_mu": cfg.noise_update_mu,
        "vad_threshold_db": cfg.vad_threshold_db,
        "vad_hangover_frames": cfg.vad_hangover_frames,
        "adaptive_alpha": cfg.adaptive_alpha,
        "alpha_snr_ref_db": cfg.alpha_snr_ref_db,
        "alpha_snr_slope": cfg.alpha_snr_slope,
        "alpha_min": cfg.alpha_min,
        "alpha_max": cfg.alpha_max,
        "high_snr_bypass_db": cfg.high_snr_bypass_db,
        "noise_leader_blocks": cfg.noise_frames + 1,
        "ceiling_ms": pipeline.ceiling_ms,
        "preferred_ms": pipeline.preferred_ms,
        "snr_db": args.snr_db, "signal_source": sig_src,
        "corpus_name": args.corpus_name,
        "duration_s": args.duration_s,
    }
    return BenchmarkResult(
        config=config_dict, power=power, latency=report,
        stoi_input=stoi_in, stoi_output=stoi_out,
        segsnr_input=seg_in, segsnr_output=seg_out,
        signal_note=signal_note,
    )


def _reconstruct_output(pipeline: NoiseReductionPipeline,
                        noisy: np.ndarray) -> np.ndarray:
    """Re-run the same stream to capture the actual output signal.

    run_stream() discards the samples; for metrics we need them. Re-running
    on the identical stream after reset() is deterministic (identical input
    after reset() gives identical output - stated contract). The latency
    report was already snapshotted by run_stream() before this call, so
    resetting the pipeline for replay is safe.
    """
    pipeline.reset()
    return pipeline.process(noisy)


def render_report(res: BenchmarkResult) -> str:
    lines = []
    lines.append("Decibel Silicon - DSP benchmark (Blueprint numbers)")
    lines.append("=" * 64)
    lines.append("config:")
    for k, v in res.config.items():
        lines.append(f"  {k:16s}: {v}")
    lines.append("-" * 64)
    lines.append("POWER DRAW ESTIMATE (model, not a measurement)")
    lines.append(f"  {res.power['value_mw'] * 1000.0:.2f} uW  "
                 f"({res.power['value_mw']:.4f} mW)")
    lines.append(f"  source     : {res.power['source']} - {res.power['reference']}")
    lines.append(f"  cross-check: effective {res.power['effective_mhz']:.2f} MHz x "
                 f"{res.power['crosscheck_uW_per_MHz_range']} uW/MHz (65nm, "
                 f"cited range) = "
                 f"{res.power['effective_mhz'] * res.power['crosscheck_uW_per_MHz_range'][0]:.3f}.."
                 f"{res.power['effective_mhz'] * res.power['crosscheck_uW_per_MHz_range'][1]:.3f} uW")
    if "bypass_value_mw" in res.power:
        lines.append("  high-SNR bypass (revision 2.1):")
        lines.append(f"    threshold        : {res.config['high_snr_bypass_db']:.1f} dB")
        lines.append(f"    bypassed frames  : {res.power['bypassed_frames']}/"
                     f"{res.power['active_frames']} ACTIVE (identity output)")
        lines.append(f"    per-frame when bypassed: "
                     f"{res.power['bypass_cycles_per_frame']:.0f} cycles -> "
                     f"{res.power['bypass_value_mw'] * 1000.0:.2f} uW "
                     f"(subtraction + alpha map skipped; meters + log2 kept)")
        lines.append("    NOTE: the stream draws between the two values; the "
                     "headline is the all-subtract worst case.")
    lines.append(f"  caveat     : {res.power['caveat']}")
    lines.append("-" * 64)
    lat = res.latency
    lines.append("LATENCY (measured, per frame vs ceiling)")
    lines.append(f"  measured algorithmic delay : {lat.measured_algo_delay_ms:.2f} ms "
                 f"(impulse probe; window-center convention)")
    lines.append(f"  ceiling / preferred        : {lat.ceiling_ms:.1f} / "
                 f"{lat.preferred_ms:.1f} ms")
    lines.append(f"  frames {lat.passed_frames}/{lat.total_frames} passed ceiling "
                 f"-> overall {('PASS' if lat.passed_ceiling else 'FAIL')}")
    lines.append(f"  worst 5 frames (host compute only, dev machine):")
    for f in sorted(lat.frames, key=lambda f: f.compute_ms, reverse=True)[:5]:
        lines.append(f"    frame {f.frame_index:3d}: algo {f.algo_delay_ms:6.2f} ms "
                     f"[{'PASS' if f.passed_ceiling else 'FAIL'}] "
                     f"compute {f.compute_ms:8.3f} ms (host only) "
                     f"[budget {'PASS' if f.passed_budget else 'FAIL(host)'}]")
    lines.append("-" * 64)
    lines.append("INTELLIGIBILITY PROXY")
    for name, d in (("input", res.stoi_input), ("processed", res.stoi_output)):
        v = "n/a" if d["value"] is None else f"{d['value']:.3f}"
        extra = "" if d.get("reason") is None else f"  ({d['reason']})"
        lines.append(f"  STOI   {name:9s}: {v}   {d['label']}{extra}")
    for name, d in (("input", res.segsnr_input), ("processed", res.segsnr_output)):
        v = "n/a" if np.isnan(d["value"]) else f"{d['value']:.1f} dB"
        lines.append(f"  segSNR {name:9s}: {v}   {d['label']}")
    lines.append("  deltas (processed - input):")
    si = res.stoi_input["value"]; so = res.stoi_output["value"]
    if si is not None and so is not None:
        lines.append(f"    dSTOI   = {so - si:+.3f}")
    lines.append(f"    dsegSNR = {res.segsnr_output['value'] - res.segsnr_input['value']:+.1f} dB")
    lines.append("  ALGORITHM: adaptive revision 2 - VAD-gated noise tracking +")
    lines.append("  SNR-adaptive over-subtraction factor alpha_eff (see")
    lines.append("  docs/research/noise-reduction.md). The v1 static-estimate")
    lines.append("  baseline is preserved behind config flags (noise_tracking=0,")
    lines.append("  alpha_snr_slope=0) and its honest finding remains recorded in")
    lines.append("  the docs. Whether the numbers above improve on v1 is the")
    lines.append("  sweep comparison in docs/research/benchmark-methodology.md.")
    if res.config.get("high_snr_bypass_db"):
        lines.append("  High-SNR bypass is ENABLED: frames above the threshold pass")
        lines.append("  through with gain=1 (exact identity), closing the residual")
        lines.append("  high-SNR segSNR loss (revision 2.1).")
    lines.append(f"  caveat     : {res.signal_note}")
    lines.append("=" * 64)
    return "\n".join(lines)


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DSP benchmark and emit the three Blueprint numbers",
    )
    parser.add_argument("--fs", type=int, default=DEFAULT_FS)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP)
    parser.add_argument("--alpha", type=_positive_float, default=DEFAULT_ALPHA)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    parser.add_argument("--noise-frames", type=int, default=DEFAULT_NOISE_FRAMES)
    # v2 adaptive parameters (defaults mirror the validated algorithm).
    parser.add_argument("--noise-tracking", action="store_true",
                        default=DEFAULT_NOISE_TRACKING,
                        help="VAD-gated recursive noise update in ACTIVE")
    parser.add_argument("--noise-update-mu", type=float,
                        default=DEFAULT_NOISE_UPDATE_MU,
                        help="noise-tracking EMA coefficient in [0,1)")
    parser.add_argument("--vad-threshold-db", type=float,
                        default=DEFAULT_VAD_THRESHOLD_DB)
    parser.add_argument("--vad-hangover-frames", type=int,
                        default=DEFAULT_VAD_HANGOVER_FRAMES)
    parser.add_argument("--adaptive-alpha", action="store_true",
                        default=DEFAULT_ADAPTIVE_ALPHA,
                        help="SNR-adaptive over-subtraction factor")
    parser.add_argument("--alpha-snr-ref-db", type=float,
                        default=DEFAULT_ALPHA_SNR_REF_DB)
    parser.add_argument("--alpha-snr-slope", type=float,
                        default=DEFAULT_ALPHA_SNR_SLOPE)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument(
        "--high-snr-bypass-db", type=float, default=DEFAULT_HIGH_SNR_BYPASS_DB,
        help="revision 2.1: pass ACTIVE frames whose SNR meter exceeds this "
             "threshold through EXACTLY (gain=1, WOLA identity) and skip "
             "subtraction. None/disabled by default; recommended >= 10 dB.")
    parser.add_argument("--ceiling-ms", type=_positive_float,
                        default=DEFAULT_CEILING_MS)
    parser.add_argument("--preferred-ms", type=_positive_float,
                        default=DEFAULT_PREFERRED_MS)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--duration-s", type=_positive_float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", type=Path, default=None,
                        help="optional real clean-speech wav (mono)")
    parser.add_argument("--noise", type=Path, default=None,
                        help="optional noise wav (mono; must match --clean fs)")
    parser.add_argument("--corpus-name", type=str, default=None,
                        help="label the wav corpus for honest provenance, e.g. "
                             "'LibriSpeech + DEMAND (CC-BY 4.0)' - surfaces in "
                             "the metric labels and caveat")
    parser.add_argument("--json", type=Path, default=None,
                        help="write machine-readable labeled result here")
    parser.add_argument(
        "--sweep", action="append", default=[], metavar="KEY:V1,V2,...",
        help="repeatable cartesian sweep, e.g. "
             "--sweep snr-db:0,5,10 --sweep noise-update-mu:0,0.1,0.15. "
             "Recognized keys: snr-db, noise-update-mu, alpha-snr-slope, "
             "vad-threshold-db. Runs each cell and prints a compact table "
             "(plus --json of the last single-key JSON shape is NOT written "
             "in sweep mode; use --sweep-json for the table).",
    )
    parser.add_argument("--sweep-json", type=Path, default=None,
                        help="sweep mode: write the full table as JSON here")
    args = parser.parse_args(argv)

    if (args.clean is None) != (args.noise is None):
        parser.error("--clean and --noise must be supplied together")
    if args.corpus_name and (args.clean is None or args.noise is None):
        parser.error("--corpus-name requires --clean and --noise")

    if args.sweep:
        return _run_sweep(parser, args)

    try:
        res = run_benchmark(args)
    except (ValueError, RuntimeError) as exc:
        # Fail-closed: no numbers emitted on any validation/processing error.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(render_report(res))
    if args.json:
        args.json.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"\nwrote labeled JSON result to {args.json}")
    return 0


# --------------------------------------------------------------------------
# Cartesian sweep of adaptive parameters (extended benchmark tooling).
# Each cell runs the SAME stream builder as the single run, so sweep numbers
# are directly comparable to the single-run headline and to the v1 sweep.
# --------------------------------------------------------------------------

_SWEEP_KEYS = {
    "snr-db": "snr_db",
    "noise-update-mu": "noise_update_mu",
    "alpha-snr-slope": "alpha_snr_slope",
    "vad-threshold-db": "vad_threshold_db",
}


def _run_sweep(parser: argparse.ArgumentParser,
               args: argparse.Namespace) -> int:
    import copy

    cells: list[tuple[str, list[float]]] = []
    for spec in args.sweep:
        key, _, vals_s = spec.partition(":")
        if key not in _SWEEP_KEYS:
            parser.error(f"--sweep key must be one of {sorted(_SWEEP_KEYS)}, "
                         f"got {key!r}")
        try:
            vals = [float(v) for v in vals_s.split(",") if v.strip()]
        except ValueError:
            parser.error(f"--sweep {key} values must be floats, got {vals_s!r}")
        if not vals:
            parser.error(f"--sweep {key} needs at least one value")
        cells.append((key, vals))

    rows = [{}]
    for key, vals in cells:
        rows = [dict(r, **{_SWEEP_KEYS[key]: v}) for r in rows for v in vals]

    table = []
    headers = (["snr_db", "noise_update_mu", "alpha_snr_slope",
                "vad_threshold_db"] +
               ["stoi_in", "stoi_out", "dSTOI", "seg_in_dB", "seg_out_dB",
                "dseg_dB", "delay_ms", "power_uW"])
    for over in rows:
        cell_args = copy.deepcopy(args)
        for attr, val in over.items():
            setattr(cell_args, attr, val)
        try:
            res = run_benchmark(cell_args)
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR (sweep cell {over}): {exc}", file=sys.stderr)
            return 2
        si = res.stoi_input["value"]
        so = res.stoi_output["value"]
        d_stoi = "n/a" if (si is None or so is None) else f"{so - si:+.3f}"
        table.append({
            "snr_db": over.get("snr_db", args.snr_db),
            "noise_update_mu": over.get("noise_update_mu", args.noise_update_mu),
            "alpha_snr_slope": over.get("alpha_snr_slope", args.alpha_snr_slope),
            "vad_threshold_db": over.get("vad_threshold_db", args.vad_threshold_db),
            "stoi_in": None if si is None else round(float(si), 4),
            "stoi_out": None if so is None else round(float(so), 4),
            "dSTOI": d_stoi if isinstance(d_stoi, str) else round(d_stoi, 4),
            "seg_in_dB": round(float(res.segsnr_input["value"]), 2),
            "seg_out_dB": round(float(res.segsnr_output["value"]), 2),
            "dseg_dB": round(float(res.segsnr_output["value"]
                                   - res.segsnr_input["value"]), 2),
            "delay_ms": round(float(res.latency.measured_algo_delay_ms), 3),
            "power_uW": round(float(res.power["value_mw"]) * 1000.0, 2),
        })

    out = []
    out.append("Decibel Silicon - DSP adaptive sweep (Blueprint numbers)")
    out.append("=" * 118)
    out.append("  " + "  ".join(f"{h:>14}" for h in headers))
    for row in table:
        vals = [f"{row['snr_db']:>14.1f}",
                f"{row['noise_update_mu']:>14.3f}",
                f"{row['alpha_snr_slope']:>14.3f}",
                f"{row['vad_threshold_db']:>14.1f}",
                f"{row['stoi_in'] if row['stoi_in'] is not None else 'n/a':>14}",
                f"{row['stoi_out'] if row['stoi_out'] is not None else 'n/a':>14}",
                f"{row['dSTOI']:>14}",
                f"{row['seg_in_dB']:>14.2f}",
                f"{row['seg_out_dB']:>14.2f}",
                f"{row['dseg_dB']:>14.2f}",
                f"{row['delay_ms']:>14.3f}",
                f"{row['power_uW']:>14.2f}"]
        out.append("  " + "  ".join(vals))
    out.append("  labels: STOI/segSNR = measured proxies on synthetic signal; "
               "delay = measured (probe);")
    out.append("  power = model (cite), not measured. Full methodology: "
               "docs/research/benchmark-methodology.md")
    out.append("=" * 118)
    text = "\n".join(out)
    print(text)
    if args.sweep_json:
        args.sweep_json.write_text(json.dumps(
            {"headers": headers, "rows": table}, indent=2))
        print(f"\nwrote sweep table JSON to {args.sweep_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())