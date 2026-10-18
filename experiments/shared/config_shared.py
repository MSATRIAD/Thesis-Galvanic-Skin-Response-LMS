"""
Konfigurasi BERSAMA untuk semua eksperimen arsitektur.

Berisi keputusan level-DATA yang HARUS identik antar-arsitektur agar
perbandingan adil: sumber data, sampling, windowing, skema label, evaluasi.
Parameter level-arsitektur (representasi, model, training) ada di
masing-masing folder eksperimen (exp*/config.py).
"""

import os
import torch
from pathlib import Path

# Root repo = dua tingkat di atas file ini.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder data mentah (`data pc 1`, `data pc 2`, dst.) berada langsung di root repo.
# Pemilihan berkas CSV sensornya dilakukan oleh `shared.data.list_files()` agar
# tabel fitur dan tabel NASA-TLX yang juga berformat CSV tidak ikut terbaca.
DATA_DIR = str(PROJECT_ROOT)
WWL_CSV = str(PROJECT_ROOT / "nasa_tlx_results" / "summary_wwl.csv")
DETAIL_CSV = str(PROJECT_ROOT / "nasa_tlx_results" / "detail_all.csv")  # rating per dimensi
SHARED_CACHE = str(PROJECT_ROOT / "experiments" / "_cache_shared")

# ---------------------------------------------------------------- Sinyal GSR
FS = 10                          # target sampling rate (Hz) setelah resample uniform
GSR_COLUMN = "gsrRaw"
TIME_COLUMN = "TimeStamp"
ACTIVITY_CODE_COLUMN = "Activity_Code"

# Activity_Code -> kolom di summary_wwl.csv (0=calibration/resting/nasa_tlx -> dibuang)
ACTIVITY_CODE_MAP = {1: "watching", 2: "reading", 3: "discussion", 4: "quiz"}
EXCLUDE_ACTIVITY_CODES = [0]

# ---------------------------------------------------------------- Windowing
WINDOW_SEC = 30
HOP_SEC = 15
WINDOW_SIZE = int(FS * WINDOW_SEC)   # 300 sampel
HOP_SIZE = int(FS * HOP_SEC)         # 150 sampel

# ---------------------------------------------------------------- Preprocessing EDA (bersama)
MAX_INTERP_SEC = 2.0     # gap dropout > ini -> window dibuang (bukan diinterpolasi)
SMOOTH_CUTOFF = 1.0      # low-pass (Hz) untuk buang derau di atas pita EDA
# Penyamaan baseline per-subjek: 'none' | 'range' (Lykken) | 'zscore'
BASELINE_NORM = "zscore"

# ---------------------------------------------------------------- Pelabelan
# 'extreme'  : tertil (<=P33 Low, >=P67 High, buang tengah)  [UTAMA]
# 'median'   : median split
# 'category' : ambang tetap kategori NASA-TLX (High = WWL >= CATEGORY_HIGH_MIN)
#              CATEGORY_LEVEL='subject' (rata-rata per responden) | 'activity' (per aktivitas)
LABEL_STRATEGY = "extreme"
EXTREME_LOW_Q = 0.33
EXTREME_HIGH_Q = 0.67

# Kategori NASA-TLX: 1 Low 0-9 | 2 Medium 10-29 | 3 Somewhat High 30-49
#                    4 High 50-79 | 5 Very High 80-100  -> biner: High jika WWL>=50
CATEGORY_HIGH_MIN = 50.0
CATEGORY_LEVEL = "subject"   # 'subject' (rata-rata responden) | 'activity'

# ---------------------------------------------------------------- Evaluasi
RANDOM_SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

os.makedirs(SHARED_CACHE, exist_ok=True)
