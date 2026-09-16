"""Fixed experiment constants for the learned gain model (dsp/learned).

The held-out category and the training categories are the ONLY source of
truth for the train/holdout split. The data loader asserts it never sees the
held-out category; moving a category here after training starts is a protocol
violation, not a config change.
"""

# Held-out noise category (designated in dsp/corpus/README.md BEFORE any
# training code ran). It never appears in training or validation manifests;
# it is evaluated exactly once, at the end of the Task 4 table.
HELD_OUT_CATEGORY = "PCAFETER"

# Training noise categories (diverse stationarity: domestic/quiet,
# street/open-air bursty, transportation non-stationary).
TRAIN_CATEGORIES = ("DKITCHEN", "STRAFFIC", "TMETRO")

# Channels: training draws from TRAIN_CHANNELS; eval slices in dsp/corpus
# come from different channels so eval numbers can never reflect the exact
# training audio (no train/eval intra-category overlap).
TRAIN_CHANNELS = ("ch01", "ch05", "ch09", "ch13")

# SNR range matched to the classical benchmark sweep.
SNRS_DB = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

# STFT geometry - MUST match the classical pipeline (128-pt rFFT, hop 64).
FS = 16000
N_FFT = 128
HOP = 64

# Mel band count (RNNoise's insight is band-level gains, not its exact 22
# bands; 24 mel bands cover 0-8 kHz on our 65-bin grid - see banding.py).
N_BANDS = 24

# GRU size: 32 hidden units on 48 features (24 mixture + 24 noise bands).
# Kept at the size the power manifest in power_model.py counts MACs for.
HIDDEN = 32

# Determinism seed for all data shuffling and training init.
SEED = 20260916