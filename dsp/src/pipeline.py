"""Streaming harness + latency measurement for the noise-reduction pipeline.

Latency contract (this module MEASURES, it does not assert):
  - ``measure_algorithmic_delay_ms()`` runs an impulse probe through the full
    WOLA chain and reports the structural blocking delay of this exact
    implementation (buffer + window + OLA). The number is derived from a
    running stream, not taken from the literature.
  - ``process_frame()`` records, for every frame, the host compute time and a
    PASS/FAIL verdict against the Blueprint ceiling (10-20 ms) and the
    per-frame real-time budget (hop/fs).

Provenance rules:
  - ``algo_delay_ms`` is the measured structural delay (constant in steady
    state; verified by the probe).
  - ``compute_ms`` is DEV-MACHINE-ONLY time (Python/numpy on this host). It is
    never presented as target-silicon execution time. The budget check
    (frame_budget_ms) is a TARGET-only check: the Python prototype is
    expected to fail it; only the fixed-point firmware build is allowed to
    claim real-time per-frame compliance.
  - A frame fails iff algo_delay_ms > ceiling_ms (the number the pitch cares
    about). The budget flag is reported separately and never folded into the
    ceiling verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .spectral_subtraction import (
    SpectralSubtraction,
    SpectralSubtractionConfig,
)

# Blueprint envelope (docs/research/latency-budget.md)
DEFAULT_CEILING_MS = 20.0
DEFAULT_PREFERRED_MS = 10.0


@dataclass
class FrameResult:
    """Per-frame latency record."""

    frame_index: int
    algo_delay_ms: float      # measured structural delay (probe-verified)
    compute_ms: float         # host compute time (DEV-MACHINE-ONLY)
    ceiling_ms: float
    passed_ceiling: bool      # the pitch-relevant verdict
    frame_budget_ms: float    # hop/fs target budget (TARGET-only check)
    passed_budget: bool       # expected to FAIL on the Python prototype

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed_ceiling else "FAIL"


@dataclass
class LatencyReport:
    """Aggregate latency statistics + per-frame rows."""

    frames: list[FrameResult] = field(default_factory=list)
    measured_algo_delay_ms: float = 0.0
    ceiling_ms: float = DEFAULT_CEILING_MS
    preferred_ms: float = DEFAULT_PREFERRED_MS

    @property
    def passed_ceiling(self) -> bool:
        return bool(self.frames) and all(f.passed_ceiling for f in self.frames)

    @property
    def passed_frames(self) -> int:
        return sum(1 for f in self.frames if f.passed_ceiling)

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def worst_compute_ms(self) -> float:
        return max((f.compute_ms for f in self.frames), default=0.0)

    @property
    def p99_compute_ms(self) -> float:
        if not self.frames:
            return 0.0
        vals = sorted(f.compute_ms for f in self.frames)
        return float(vals[min(len(vals) - 1, int(0.99 * len(vals)))])

    def summary_lines(self) -> list[str]:
        lines = [
            f"measured algorithmic delay   : {self.measured_algo_delay_ms:7.2f} ms "
            f"(impulse probe)",
            f"ceiling (Blueprint)          : {self.ceiling_ms:7.2f} ms"
            + (f"  [preferred: {self.preferred_ms:.1f} ms]" if self.preferred_ms else ""),
            f"frames                       : {self.total_frames:6d}  "
            f"(passed {self.passed_frames})",
            f"overall ceiling verdict      : {'PASS' if self.passed_ceiling else 'FAIL'}",
            f"host compute  p99 / worst    : {self.p99_compute_ms:7.3f} / "
            f"{self.worst_compute_ms:7.3f} ms  (DEV-MACHINE-ONLY, not target)",
        ]
        return lines


class NoiseReductionPipeline:
    """Streaming noise-reduction harness with latency instrumentation.

    Sequential-streaming contract (Phase 3): one consumer per instance; call
    reset() to begin a new stream; identical input after reset() gives
    identical output.
    """

    def __init__(
        self,
        *,
        config: SpectralSubtractionConfig | None = None,
        gain_hook=None,
        ceiling_ms: float = DEFAULT_CEILING_MS,
        preferred_ms: float = DEFAULT_PREFERRED_MS,
    ) -> None:
        if not (0 < ceiling_ms <= 1000):
            raise ValueError(f"ceiling_ms must be in (0, 1000], got {ceiling_ms!r}")
        self.config = config or SpectralSubtractionConfig()
        if ceiling_ms < preferred_ms:
            raise ValueError("ceiling_ms must be >= preferred_ms")
        self.ceiling_ms = float(ceiling_ms)
        self.preferred_ms = float(preferred_ms)
        # Learned-model bridge (dsp/learned): same interface point as the
        # classical gain decision; None keeps the pipeline byte-identical.
        self._ss = SpectralSubtraction(self.config, gain_hook=gain_hook)
        self._frame_results: list[FrameResult] = []
        self._measured_delay_ms: float | None = None
        self.reset()

    # -- stream control -----------------------------------------------------

    def reset(self) -> None:
        self._ss.reset()
        self._frame_results = []
        self._measured_delay_ms = None

    @property
    def state(self) -> str:
        return self._ss.state

    @property
    def bypass_count(self) -> int:
        """ACTIVE frames emitted on the high-SNR identity path (revision 2.1);
        zero when the bypass is disabled."""
        return self._ss.bypass_count

    @property
    def active_frames(self) -> int:
        """Frames processed in ACTIVE state (the denominator for bypass_count
        when reporting the bypassed fraction)."""
        return self._ss.active_frames

    @property
    def frame_budget_ms(self) -> float:
        return self.config.hop / self.config.fs * 1000.0

    # -- processing ---------------------------------------------------------

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Feed arbitrary-length samples; return processed samples of equal length.

        Uses process_frame() internally so every emitted chunk carries a
        FrameResult. The final partial frame is zero-padded for the internal
        processing step but its output is truncated back to the exact input
        length (the pad itself is never returned to the caller).
        """
        samples = np.asarray(samples, dtype=np.float64)
        n = samples.size
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, self.config.hop):
            chunk = samples[start : start + self.config.hop]
            if chunk.size < self.config.hop:
                # Pad the tail with zeros to complete the final frame (stream
                # end semantics). Zero-padding is explicit, never implicit.
                chunk = np.concatenate([chunk, np.zeros(self.config.hop - chunk.size)])
            block, _ = self.process_frame(chunk)
            keep = min(self.config.hop, n - start)
            out[start : start + keep] = block[:keep]
        return out

    def process_frame(self, hop_chunk: np.ndarray) -> tuple[np.ndarray, FrameResult]:
        """Consume exactly `hop` samples; return (output_block, FrameResult)."""
        if self._measured_delay_ms is None:
            # Lazy one-time structural measurement on the live configuration.
            self._measured_delay_ms = self.measure_algorithmic_delay_ms()

        t0 = time.perf_counter()
        out = self._ss.process_frame(hop_chunk)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        idx = len(self._frame_results)
        res = FrameResult(
            frame_index=idx,
            algo_delay_ms=self._measured_delay_ms,
            compute_ms=compute_ms,
            ceiling_ms=self.ceiling_ms,
            passed_ceiling=self._measured_delay_ms <= self.ceiling_ms,
            frame_budget_ms=self.frame_budget_ms,
            passed_budget=compute_ms <= self.frame_budget_ms,
        )
        self._frame_results.append(res)
        return out, res

    # -- measurement --------------------------------------------------------

    def measure_algorithmic_delay_ms(self, impulse_gain: float = 0.8,
                                     threshold: float = 1e-3) -> float:
        """Impulse-probe the structural delay of this exact WOLA chain.

        Feeds a mid-window impulse through a fresh stream and measures the
        output sample index (relative to the impulse's input index) at which
        the response first exceeds `threshold`. The result is the measured
        blocking delay of this implementation, NOT a literature value.

        The probe shares the pipeline's configuration (same n_fft/hop/alpha/
        floor) but uses a temporary internal instance so the caller's stream
        state is untouched.
        """
        cfg = self.config
        probe = SpectralSubtraction(cfg)
        hop = cfg.hop
        n_fft = cfg.n_fft

        # Warm the OLA state (fill one full window's worth of silence) so the
        # impulse is measured in steady-state positioning.
        warm_blocks = (n_fft // hop) + 2
        for _ in range(warm_blocks):
            probe.process_frame(np.zeros(hop, dtype=np.float64))

        # Place the impulse at the START of the next frame's input. In this
        # WOLA the buffer is the newest N samples centered on the frame
        # boundary, so the impulse lands at window position n_fft//2 = hop
        # (the periodic-Hann peak); its response exits hop samples later -
        # this is the uniform structural group delay of the chain.
        offset_blocks = (n_fft // 2) // hop  # 1 block -> impulse at window pos 64
        for _ in range(offset_blocks):
            probe.process_frame(np.zeros(hop, dtype=np.float64))
        impulse_global_idx = (warm_blocks + offset_blocks) * hop
        # The probe needs the impulse to land at the window peak (pos n_fft/2).
        # With hop == n_fft there is no overlap position to place it at; that
        # degenerate WOLA (no overlap) is not measurable by this probe.
        if offset_blocks * hop != n_fft // 2:
            raise ValueError(
                "impulse probe requires hop < n_fft (overlap); "
                f"got hop={hop}, n_fft={n_fft}"
            )

        out_blocks: list[np.ndarray] = []
        frame = np.zeros(hop, dtype=np.float64)
        frame[0] = impulse_gain
        out_blocks.append(probe.process_frame(frame))
        # Feed enough silence to flush the impulse through the OLA tail.
        for _ in range(n_fft // hop):
            out_blocks.append(probe.process_frame(np.zeros(hop, dtype=np.float64)))

        emitted = np.concatenate(out_blocks)
        above = np.where(np.abs(emitted) > threshold)[0]
        if above.size == 0:
            raise RuntimeError(
                "impulse probe found no response above threshold; "
                "pipeline is not reconstructing - do not trust latency numbers"
            )
        # emitted[] is relative to the base output index; convert to absolute.
        base_output_idx = (warm_blocks + offset_blocks) * hop
        first_response_idx = base_output_idx + int(above[0])
        delay_samples = first_response_idx - impulse_global_idx
        delay_ms = delay_samples / cfg.fs * 1000.0
        return delay_ms

    # -- reporting ----------------------------------------------------------

    def report(self) -> LatencyReport:
        return LatencyReport(
            frames=list(self._frame_results),
            measured_algo_delay_ms=(
                self._measured_delay_ms
                if self._measured_delay_ms is not None
                else self.measure_algorithmic_delay_ms()
            ),
            ceiling_ms=self.ceiling_ms,
            preferred_ms=self.preferred_ms,
        )

    def print_report(self) -> str:
        report = self.report()
        lines = report.summary_lines()
        text = "\n".join(lines)
        print(text)
        return text


def run_stream(pipeline: NoiseReductionPipeline, samples: np.ndarray) -> LatencyReport:
    """Feed a stream and return the latency report.

    All frames are included in the report. Metrics consumers (e.g. the
    benchmark) are responsible for excluding the warm-up passthrough period
    from *quality* metrics (STOI/segSNR windows); latency rows are valid for
    every frame because the structural delay and host compute are recorded
    regardless of processing state.
    """
    pipeline.reset()
    samples = np.asarray(samples, dtype=np.float64)
    chunks = [
        samples[i : i + pipeline.config.hop]
        for i in range(0, samples.size, pipeline.config.hop)
    ]
    for chunk in chunks:
        if chunk.size < pipeline.config.hop:
            chunk = np.concatenate(
                [chunk, np.zeros(pipeline.config.hop - chunk.size)]
            )
        pipeline.process_frame(chunk)
    return pipeline.report()