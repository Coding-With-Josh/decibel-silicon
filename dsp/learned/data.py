"""Training data: LibriSpeech clean speech mixed with TRAIN noise categories.

INTEGRITY RULES (Phase 3 controls, implemented here, not just documented):
  1. The train/val split is FILE-LEVEL and SPEAKER-DISJOINT: a speaker
     appears in exactly one split, so no utterance's frames can straddle the
     boundary (frame-level random splits would bias checkpoint selection).
  2. The held-out category (constants.HELD_OUT_CATEGORY) is asserted out of
     every manifest. Training can never touch it - moving it here is a
     protocol violation that raises.
  3. Determinism: all shuffles use np.random.default_rng(SEED).

Feature geometry (matches the classical pipeline): N=128, hop=64, periodic
Hann^2 window, rFFT magnitude. Per frame: (2*N_BANDS) features = log10 band
RMS of the mixture + log10 band RMS of the noise estimate, and N_BANDS
targets = ideal ratio mask (IRM) sqrt(P_s/(P_s+P_n)) per band. Training-time
"noise estimate" feature is the MEDIAN band RMS of the noise-only segment
(an utterance-constant proxy for the pipeline's leader average); a fraction
of examples is jittered per-frame (+-NOISE_FEATURE_JITTER_DB) so the model
does not over-trust a perfect oracle (eval sees a real leader estimate that
can drift on non-stationary noise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .banding import BIN_TO_BAND, band_rms
from .constants import (
    HELD_OUT_CATEGORY,
    HIDDEN,
    HOP,
    N_BANDS,
    N_FFT,
    SEED,
    SNRS_DB,
    TRAIN_CATEGORIES,
    TRAIN_CHANNELS,
)

NOISE_FEATURE_JITTER_DB = 1.5
JITTER_PROB = 0.5
FEATURE_EPS = 1e-8


@dataclass
class Utterance:
    """One segment of clean speech (LibriSpeech file)."""

    file: Path
    speaker: str
    samples: np.ndarray  # float64 [-1, 1], fs=16000

    @property
    def seconds(self) -> float:
        return self.samples.size / 16000.0


@dataclass
class MixRecord:
    """Deterministic mix + precomputed features/targets for one (utt, snr, cat)."""

    key: str
    speaker: str
    category: str
    snr_db: float
    features: np.ndarray   # (T, 2*N_BANDS) float32
    targets: np.ndarray    # (T, N_BANDS) float32
    n_frames: int


def corpus_root() -> Path:
    return Path(__file__).resolve().parent.parent / "corpus"


def fetch_data_root() -> Path:
    """Location of the large fetched-but-uncommitted media (gitignored)."""
    return corpus_root() / "_fetch_demand"


def devclean_root() -> Path:
    return fetch_data_root() / "dev-clean" / "LibriSpeech" / "dev-clean"


def _periodic_hann2(n: int) -> np.ndarray:
    n_ = np.arange(n)
    return np.sin(np.pi * n_ / n) ** 2


def _band_feature(mags: np.ndarray) -> np.ndarray:
    """(T, n_bins) magnitude -> (T, N_BANDS) log10 band RMS features."""
    rms = band_rms(mags, BIN_TO_BAND, N_BANDS)
    return np.log10(np.maximum(rms, FEATURE_EPS))


def load_noise_segments(categories=TRAIN_CATEGORIES,
                        channels=TRAIN_CHANNELS) -> dict[str, dict[str, np.ndarray]]:
    """{category: {channel: mono float64 samples}} from the fetched DEMAND dirs."""
    root = fetch_data_root()
    out: dict[str, dict[str, np.ndarray]] = {}
    for cat in categories:
        out[cat] = {}
        for ch in channels:
            p = root / cat / f"{ch}.wav"
            if not p.exists():
                raise FileNotFoundError(f"missing training noise {p}")
            x, sr = sf.read(p, dtype="float64")   # normalized [-1, 1]
            if sr != 16000 or x.ndim != 1:
                raise ValueError(f"training noise must be 16k mono: {p}")
            out[cat][ch] = x
    return out


def load_speech_utterances(max_speech_seconds: float,
                           rng: np.random.Generator
                           ) -> tuple[list[Utterance], list[Utterance]]:
    """Load LibriSpeech dev-clean files up to a speech budget; speaker split.

    Returns (train_utts, val_utts). Speakers are drawn until the budget is
    reached; a fixed few speakers are set aside for validation so the model's
    val curve measures unseen speakers, never unseen frames of seen speakers.
    """
    root = devclean_root()
    if not root.exists():
        raise FileNotFoundError(
            f"LibriSpeech dev-clean not extracted at {root}; run "
            "corpus/fetch_corpus.sh or extract dev-clean.tar.gz into "
            "corpus/_fetch_demand/")
    speakers = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not speakers:
        raise FileNotFoundError(f"no speaker dirs under {root}")
    rng.shuffle(speakers)

    train_speakers: list[str] = []
    val_speakers: list[str] = []
    budget = max_speech_seconds
    # Small fixed val set (a handful of speakers, whatever their duration).
    val_speakers = speakers[-2:]
    train_speakers = speakers[:-2]

    def load_speaker(spk: str) -> list[Utterance]:
        utts = []
        for chap in (root / spk).iterdir():
            if not chap.is_dir():
                continue
            for f in sorted(chap.glob("*.flac")):
                x, sr = sf.read(f, dtype="float64")   # normalized [-1, 1]
                if sr != 16000:
                    raise ValueError(f"expected 16k flac: {f} (sr={sr})")
                x = x.astype(np.float64)
                if x.size < 2 * N_FFT:
                    continue  # too short to frame meaningfully
                utts.append(Utterance(file=f, speaker=spk, samples=x))
        return utts

    train: list[Utterance] = []
    used = 0.0
    for spk in train_speakers:
        if used >= budget:
            break
        for u in load_speaker(spk):
            train.append(u)
            used += u.seconds
            if used >= budget:
                break

    val: list[Utterance] = []
    for spk in val_speakers:
        val.extend(load_speaker(spk))

    if not train:
        raise RuntimeError("training split is empty; raise --speech-seconds")
    return train, val


def _frames(x: np.ndarray) -> np.ndarray:
    """(samples,) -> (T, n_bins) magnitude spectra using the classical
    framing/window (periodic Hann^2, N=128, hop=64)."""
    win = _periodic_hann2(N_FFT)
    n = x.size
    n_frames = max(1, (n - N_FFT) // HOP + 1)
    frames = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float64)
    for t in range(n_frames):
        seg = x[t * HOP : t * HOP + N_FFT]
        frames[t] = np.abs(np.fft.rfft(seg * win, n=N_FFT))
    return frames


def _irm_target(clean_mags: np.ndarray, noise_mags: np.ndarray) -> np.ndarray:
    """IRM per band: sqrt(P_s / (P_s + P_n)). Oracle targets, training only."""
    ps = band_rms(clean_mags, BIN_TO_BAND, N_BANDS) ** 2
    pn = band_rms(noise_mags, BIN_TO_BAND, N_BANDS) ** 2
    return np.sqrt(ps / (ps + pn + 1e-12))


def _mix_record_key(utt: Utterance, snr: float, cat: str, ch: str) -> str:
    return f"{utt.speaker}_{utt.file.stem}_{snr:+.0f}dB_{cat}_{ch}"


def _record_seed(key: str) -> int:
    """Stable 32-bit seed from a record key.

    NOT built on builtins.hash(): Python string hashing is randomized per
    process (PYTHONHASHSEED), so hash(key) would give a different mix on
    every run and the feature cache would be an unsound memoization. hashlib
    is stable across processes and interpreters.
    """
    import hashlib

    return int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")


def build_mix_record(utt: Utterance, noise: np.ndarray, snr_db: float,
                     category: str, ch: str, jitter_ok: bool) -> MixRecord:
    """Mix one utterance with a noise segment at snr_db; compute features+targets.

    noise is a full noise-channel array; a deterministic start offset (seeded
    by the record key) is drawn here so caching is a faithful memoization.
    The noise feature is the median band RMS over the noise-only frames
    (utterance-constant, a proxy for the pipeline's leader average),
    optionally jittered per-frame within +-NOISE_FEATURE_JITTER_DB.
    """
    n = utt.samples.size
    # Deterministic per-record randomness: the cache is a faithful memoization
    # - the same key ALWAYS yields the same mix/features, no matter how many
    # records were built before or after this one.
    rec_rng = np.random.default_rng(
        _record_seed(_mix_record_key(utt, snr_db, category, ch)))
    seg = noise[rec_rng.integers(0, max(1, noise.size - n)) :]
    if seg.size < n:
        seg = np.concatenate([seg, noise[: n - seg.size]])
    seg = seg[:n]

    p_clean = float(np.mean(utt.samples ** 2))
    p_noise = float(np.mean(seg ** 2)) or 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0)))
    scaled_noise = scale * seg
    mixture = utt.samples + scaled_noise

    mix_mags = _frames(mixture)
    clean_mags = _frames(utt.samples)
    noise_mags = _frames(scaled_noise)

    f_mix = _band_feature(mix_mags)
    f_noise = _band_feature(noise_mags)
    median_noise = np.median(f_noise, axis=0, keepdims=True)  # (1, B)
    if jitter_ok and rec_rng.uniform() < JITTER_PROB:
        jit = rec_rng.uniform(-NOISE_FEATURE_JITTER_DB,
                              NOISE_FEATURE_JITTER_DB,
                              size=(mix_mags.shape[0], N_BANDS))
        f_noise_feat = median_noise + jit
    else:
        f_noise_feat = np.repeat(median_noise, mix_mags.shape[0], axis=0)

    features = np.concatenate([f_mix, f_noise_feat], axis=1).astype(np.float32)
    targets = _irm_target(clean_mags, noise_mags).astype(np.float32)
    return MixRecord(
        key=_mix_record_key(utt, snr_db, category, ch),
        speaker=utt.speaker,
        category=category,
        snr_db=snr_db,
        features=features,
        targets=targets,
        n_frames=features.shape[0],
    )


def make_manifest(train_utts: list[Utterance], val_utts: list[Utterance],
                  noise: dict[str, dict[str, np.ndarray]],
                  cache_dir: Path) -> dict[str, list[MixRecord]]:
    """Build (and cache) ALL mixtures for train and val.

    Integrity: category choice is restricted to TRAIN_CATEGORIES and
    HELD_OUT_CATEGORY is asserted absent (both for the record's category
    field and by never importing it here).
    """
    if HELD_OUT_CATEGORY in TRAIN_CATEGORIES:
        raise RuntimeError("held-out category is in TRAIN_CATEGORIES: "
                           "protocol violation")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def build(utts: list[Utterance]) -> list[MixRecord]:
        records: list[MixRecord] = []
        for utt in utts:
            for snr_db in SNRS_DB:
                cat = TRAIN_CATEGORIES[int(rng.integers(0, len(TRAIN_CATEGORIES)))]
                ch = TRAIN_CHANNELS[int(rng.integers(0, len(TRAIN_CHANNELS)))]
                noise_seg = noise[cat][ch]
                key = _mix_record_key(utt, snr_db, cat, ch)
                cache = cache_dir / f"{key}.npz"
                if cache.exists():
                    d = np.load(cache)
                    records.append(MixRecord(
                        key=key, speaker=utt.speaker, category=cat,
                        snr_db=snr_db,
                        features=d["features"], targets=d["targets"],
                        n_frames=int(d["features"].shape[0])))
                else:
                    rec = build_mix_record(utt, noise_seg, snr_db, cat, ch,
                                           jitter_ok=True)
                    np.savez_compressed(cache, features=rec.features,
                                        targets=rec.targets)
                    records.append(rec)
        return records

    return {"train": build(train_utts), "val": build(val_utts)}


def manifest_stats(records: list[MixRecord]) -> dict:
    cats = {r.category for r in records}
    if HELD_OUT_CATEGORY in cats:
        raise RuntimeError("held-out category leaked into a manifest")
    total_frames = sum(r.n_frames for r in records)
    return {
        "n_records": len(records),
        "categories": sorted(cats),
        "total_frames": total_frames,
        "n_speakers": len({r.speaker for r in records}),
        "snrs": sorted({r.snr_db for r in records}),
    }