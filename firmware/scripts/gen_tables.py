#!/usr/bin/env python3
"""Generate fixed-point tables and the golden reference for the firmware port.

Source of truth: the validated (tested) floating-point reference in
dsp/src/spectral_subtraction.py. This script emits:

  src/tables.h      - periodic-Hann^2 analysis window (Q15)
                    - radix-2 twiddle factors for N=128 (Q15, k=0..N/2-1)
                    - WOLA coverage-normalization reciprocal (u32, Q15 scale)
  test/golden.h     - a deterministic 48-frame test stream (Q15) and the
                    reference pipeline's output on it (Q15) - the host test
                    compares the fixed-point port against this golden.

Everything emitted is an integer literal (no FPU on target). Regenerate with:
  make gen-tables   (uses ../dsp/.venv/bin/python)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# --- reference import (repo-relative; run from firmware/) -------------------
DSP_SRC = Path(__file__).resolve().parents[2] / "dsp" / "src"
sys.path.insert(0, str(DSP_SRC))
from spectral_subtraction import (  # noqa: E402  (the tested reference)
    SpectralSubtraction,
    SpectralSubtractionConfig,
)

FW = Path(__file__).resolve().parents[1]

N_FFT = 128
HOP = 64
NOISE_FRAMES = 8
ALPHA = 2.0
FLOOR = 0.01
FS = 16000

Q15 = 32768.0


def q15(x: float) -> int:
    """Round a float in [-1, 1) to Q15, saturating."""
    v = int(round(x * Q15))
    return max(-32768, min(32767, v))


def emit_header(path: Path, arrays: list[tuple[str, str, list[int]]],
                comment: str) -> None:
    lines = [
        "/* GENERATED FILE - do not edit by hand.",
        f" * {comment}",
        " * Regenerate:  make gen-tables",
        " */",
        "#ifndef TABLES_H",
        "#define TABLES_H",
        "",
        "#include <stdint.h>",
        "",
    ]
    for name, ctype, values in arrays:
        lines.append(f"static const {ctype} {name}[{len(values)}] = {{")
        for i in range(0, len(values), 12):
            lines.append("    " + ",".join(str(v) for v in values[i : i + 12]) + ",")
        lines.append("};")
        lines.append("")
    lines.append("#endif  /* TABLES_H */")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    # ---- tables ------------------------------------------------------------
    n = np.arange(N_FFT)
    window = np.sin(np.pi * n / N_FFT) ** 2

    tw = []
    for k in range(N_FFT // 2):
        ang = 2.0 * math.pi * k / N_FFT
        tw.append(q15(math.cos(ang)))
        tw.append(q15(math.sin(ang)))  # interleaved re, im

    cfg = SpectralSubtractionConfig(fs=FS, n_fft=N_FFT, hop=HOP,
                                    alpha=ALPHA, floor=FLOOR,
                                    noise_frames=NOISE_FRAMES)
    probe = SpectralSubtraction(cfg)
    inv_cov = probe._compute_inv_coverage()  # 1/cov[p], p in [0, hop)
    inv_cov_u32 = [int(round(c * Q15)) for c in inv_cov]

    emit_header(
        FW / "src" / "tables.h",
        [
("TWIDDLES_Q15", "int16_t",
         [int(v) for v in tw]),
            ("WIN_Q15", "int16_t", [q15(float(w)) for w in window]),
            ("INV_COV_Q15", "int32_t", inv_cov_u32),
        ],
        "twiddles/win/inv_cov for N_FFT=128, HOP=64, periodic Hann^2. "
        "Generated from dsp/src/spectral_subtraction.py (validated reference).",
    )
    print(f"wrote {FW / 'src' / 'tables.h'}")

    # ---- golden ------------------------------------------------------------
    rng = np.random.default_rng(7)
    n_frames = 48
    lead_frames = NOISE_FRAMES + 2  # 0..9 frame indices
    tone_freq = 1000.0
    t = np.arange(n_frames * HOP) / FS
    tone = 0.25 * np.sin(2 * np.pi * tone_freq * t)
    noise = 0.30 * rng.standard_normal(n_frames * HOP)
    stream = tone + noise

    # Q15-quantize the INPUT so both sides (float reference and fixed port)
    # see identical samples.
    stream_q15 = np.array([q15(float(s)) for s in stream], dtype=np.float64) / Q15
    # Force the leader to be pure noise at the same RMS as the speech-region
    # noise component (0.30) so the reference's initial noise estimate matches
    # the mixed-stream noise statistics (see dsp/benchmark.py build_test_stream).
    leader_len = lead_frames * HOP
    lead_rms = np.sqrt(np.mean(noise[:leader_len] ** 2)) or 1e-9
    stream_q15[:leader_len] = noise[:leader_len] * (0.30 / lead_rms)
    stream_q15 /= np.max(np.abs(stream_q15)) or 1.0
    stream_q15 *= 0.8

    ss = SpectralSubtraction(cfg)
    out = np.concatenate(
        [ss.process_frame(stream_q15[i : i + HOP]) for i in range(0, stream_q15.size, HOP)]
    )
    out_q15 = [q15(float(v)) for v in out]
    in_q15 = [q15(float(v)) for v in stream_q15]

    golden = [
        "/* GENERATED FILE - do not edit by hand.",
        " * Deterministic fixed-point test stream + reference (dsp/src/"
        "spectral_subtraction.py)",
        " * output. Regenerate:  make gen-tables",
        " */",
        "#ifndef GOLDEN_H",
        "#define GOLDEN_H",
        "",
        "#include <stdint.h>",
        "",
        "#define GOLDEN_N_FRAMES 48",
        "#define GOLDEN_HOP 64",
        "#define GOLDEN_N_FFT 128",
        "#define GOLDEN_NOISE_FRAMES 8",
        "#define GOLDEN_ALPHA_Q15 65536      /* 2.0 in Q15 */",
        "#define GOLDEN_FLOOR_Q15 328        /* 0.01 in Q15 */",
        "",
    ]
    for name, vals in (("GOLDEN_IN_Q15", in_q15), ("GOLDEN_OUT_Q15", out_q15)):
        golden.append(f"static const int16_t {name}[{len(vals)}] = {{")
        for i in range(0, len(vals), 12):
            golden.append("    " + ",".join(str(v) for v in vals[i : i + 12]) + ",")
        golden.append("};")
        golden.append("")
    golden.append("#endif  /* GOLDEN_H */")
    (FW / "test" / "golden.h").write_text("\n".join(golden) + "\n")
    print(f"wrote {FW / 'test' / 'golden.h'} ({len(in_q15)} samples)")


if __name__ == "__main__":
    main()