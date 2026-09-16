"""Train the small GRU gain model (offline, dev machine only).

Protocol rules enforced here:
  - The held-out category is asserted absent from every manifest.
  - Val is the ONLY signal for checkpoint selection (never the held-out set).
  - Exported artifact is float32; int8 quantization is a later, separate
    step (same sequencing discipline as the classical DSP->firmware split).
  - Deterministic: torch + numpy seeded; the cache makes reruns byte-identical
    to the first run once features exist.

Usage:
  uv run --python .venv/bin/python -m learned.train --speech-seconds 600 \
    --epochs 20 --out learned/runs/gru_v1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .banding import BIN_TO_BAND
from .constants import (
    HELD_OUT_CATEGORY,
    HIDDEN,
    N_BANDS,
    SEED,
    SNRS_DB,
    TRAIN_CATEGORIES,
)
from .data import (
    load_noise_segments,
    load_speech_utterances,
    make_manifest,
    manifest_stats,
)
from .model import GruGainNet, export_weights

CHUNK_FRAMES = 256  # truncated BPTT window (hidden carried, grads clipped at window)


def _chunk_records(records, chunk: int):
    """Yield (feat (T<=chunk,F) float32, target (T<=chunk,B) float32, n_valid)."""
    for r in records:
        t = r.features.shape[0]
        for start in range(0, t, chunk):
            end = min(start + chunk, t)
            yield r.features[start:end], r.targets[start:end], end - start, r


def _mse(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    return float(((tensor_a - tensor_b) ** 2).mean().item())


def run_epoch(model: nn.Module, records, chunk: int, optimizer=None) -> float:
    """One epoch over records. optimizer=None -> validation (no grad)."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    h: torch.Tensor | None = None
    rec_id: object = None
    for feat, target, n_valid, rec in _chunk_records(records, chunk):
        if rec.key != rec_id:
            rec_id = rec.key
            h = None
        x = torch.from_numpy(feat)[None, :, :]          # (1, T, F)
        y = torch.from_numpy(target)[None, :, :]        # (1, T, B)
        h = h.detach() if h is not None else None       # truncated BPTT
        pred, h = model(x, h)
        loss = nn.functional.mse_loss(pred[..., :n_valid, :],
                                      y[..., :n_valid, :])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * n_valid
        count += n_valid
    return total / max(1, count)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-seconds", type=float, default=600.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--chunk", type=int, default=CHUNK_FRAMES)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "_cache")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print("loading noise segments (training categories only)...")
    noise = load_noise_segments()
    print("loading LibriSpeech dev-clean ...")
    train_utts, val_utts = load_speech_utterances(args.speech_seconds, rng)
    print(f"  train utts={len(train_utts)} ({sum(u.seconds for u in train_utts):.0f} s), "
          f"val utts={len(val_utts)} ({sum(u.seconds for u in val_utts):.0f} s)")
    if HELD_OUT_CATEGORY in TRAIN_CATEGORIES:
        raise RuntimeError("held-out category in TRAIN_CATEGORIES: protocol violation")
    print("building/caching mix manifest...")
    manifest = make_manifest(train_utts, val_utts, noise, args.cache_dir)
    tr_stats = manifest_stats(manifest["train"])
    va_stats = manifest_stats(manifest["val"])
    print("  train:", tr_stats)
    print("  val  :", va_stats)
    print(f"  data ready in {time.time() - t0:.0f}s")

    n_features = 2 * N_BANDS
    model = GruGainNet(n_features=n_features, hidden=args.hidden,
                       n_bands=N_BANDS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-5)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        te = run_epoch(model, manifest["train"], args.chunk, optimizer)
        va = run_epoch(model, manifest["val"], args.chunk, None)
        scheduler.step(va)
        tag = ""
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            tag = "  <-- best val"
        print(f"epoch {epoch:2d}/{args.epochs}  train_mse={te:.4f}  "
              f"val_mse={va:.4f}{tag}", flush=True)
        if not np.isfinite(va):
            print("val loss non-finite: aborting WITHOUT export",
                  file=sys.stderr)
            return 2

    if best_state is None:
        print("no best state recorded; nothing to export",
              file=sys.stderr)
        return 2

    model.load_state_dict(best_state)
    w = export_weights(model)
    meta = {
        "n_fft": 128, "hop": 64, "fs": 16000,
        "n_bands": N_BANDS, "hidden": args.hidden,
        "n_features": n_features,
        "trained_on": ",".join(TRAIN_CATEGORIES),
        "held_out": HELD_OUT_CATEGORY,
        "snrs_db": list(SNRS_DB),
        "speech_seconds": args.speech_seconds,
        "epochs": args.epochs,
        "best_val_mse": best_val,
        "seed": args.seed,
        "total_params": w.total_params,
        "train_stats": tr_stats,
        "val_stats": va_stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        w_ir=w.w_ir, w_hr=w.w_hr, b_ih=w.b_ih, b_hh=w.b_hh,
        out_w=w.out_w, out_b=w.out_b,
        band_map=BIN_TO_BAND, meta=json.dumps(meta))
    print(f"exported float32 weights: {args.out}  ({w.total_params} params)")
    print(f"best_val_mse={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())