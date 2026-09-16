"""Learned gain-model package (offline training + numpy inference).

Layout:
    constants.py  every protocol constant (held-out split, geometry, seed)
    banding.py    mel band map shared with the pipeline
    data.py       corpus loading, deterministic mixing, IRM targets, cache
    model.py      torch training net + float32 export + numpy GRU forward
    infer.py      schema-validated npz loading + GainModelHook (pipeline bridge)
    train.py      CLI trainer (CPU, offline)
"""

from .constants import (
    HELD_OUT_CATEGORY,
    HIDDEN,
    N_BANDS,
    SEED,
    SNRS_DB,
    TRAIN_CATEGORIES,
    TRAIN_CHANNELS,
)
from .infer import GainModelHook, LoadedGainModel, load_model_npz
from .model import GruGainNet, LearnedWeights, export_weights, gru_forward_numpy

__all__ = [
    "HELD_OUT_CATEGORY",
    "HIDDEN",
    "N_BANDS",
    "SEED",
    "SNRS_DB",
    "TRAIN_CATEGORIES",
    "TRAIN_CHANNELS",
    "GruGainNet",
    "LearnedWeights",
    "export_weights",
    "gru_forward_numpy",
    "GainModelHook",
    "LoadedGainModel",
    "load_model_npz",
]