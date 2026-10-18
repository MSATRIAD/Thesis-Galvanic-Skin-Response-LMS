"""Konfigurasi spesifik arsitektur exp01: GLCT heatmap -> 2D CNN."""

import os
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(EXP_DIR / "_cache")
RESULTS_DIR = str(EXP_DIR / "results")

# ----- Representasi: HPF + GLCT -----
HPF_CUTOFF = 0.05
HPF_ORDER = 2
# Normalisasi sinyal sebelum GLCT:
#   'none'     : amplitudo mentah dipertahankan
#   'subject'  : z-score per subjek (CATATAN: menghapus baseline antar-subjek)
#   'window'   : z-score per window
SIGNAL_NORM = "none"

GLCT_N_FREQS = 64
GLCT_WINDOW_LENGTH = 64
GLCT_HOP_LENGTH = 16
GLCT_CHIRP_RATES = 11
GLCT_CHIRP_RANGE = (-5, 5)

# Normalisasi heatmap (untuk CNN):
#   'global'   : skala log1p dgn min-max GLOBAL (pertahankan beda amplitudo antar-window)
#   'per_image': min-max tiap heatmap (menghapus amplitudo absolut)
HEATMAP_NORM = "global"
CNN_INPUT_SIZE = (128, 128)

# ----- Training -----
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 6

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
