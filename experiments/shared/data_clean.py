"""Loader window GSR BEBAS KEBOCORAN NORMALISASI (berkas BARU; shared/data.py lama utuh).

Perbedaan dari shared/data.py:
  data.py  : statistik z-score per-subjek dihitung HANYA dari aktivitas berlabel
             (yang lolos EGA). Pemilihannya bergantung WWL -> label bocor ke fitur.
  data_clean.py : statistik dihitung dari SELURUH aktivitas subjek (keempatnya),
             TANPA melihat label sama sekali. Window dikeluarkan untuk semua
             aktivitas; label ditetapkan TERPISAH oleh pemanggil (dari WWL).
             -> normalisasi TAK bergantung label = bebas bocor by construction.

Output build_windows_clean():
    Xraw   : (N, WINDOW) float32   window ternormalisasi (bebas bocor)
    groups : (N,) int64            indeks subjek
    meta   : list[str]             "subjek|aktivitas" per window
    subjects : list[str]
"""
import glob
import os

import numpy as np
import pandas as pd

from .config_shared import (DATA_DIR, FS, GSR_COLUMN, TIME_COLUMN,
                            ACTIVITY_CODE_COLUMN, ACTIVITY_CODE_MAP,
                            MAX_INTERP_SEC, SMOOTH_CUTOFF, BASELINE_NORM)
from .data import subject_from_path, _resample_uniform
from . import eda_preprocessing as E


def build_windows_clean(window_sec=60, hop_sec=30, use_nk=True, verbose=False):
    WS, HS = int(FS * window_sec), int(FS * hop_sec)
    files = [f for f in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
             if not os.path.basename(f).startswith("_")]

    Xraw, groups, meta, subjects = [], [], [], []
    for f in files:
        sid = subject_from_path(f)
        df = pd.read_csv(f)
        ts = pd.to_datetime(df[TIME_COLUMN]); ts = (ts - ts.iloc[0]).dt.total_seconds().values
        codes = df[ACTIVITY_CODE_COLUMN].values
        gsr = df[GSR_COLUMN].values.astype(float)

        # 1) bersihkan SETIAP aktivitas (TANPA melihat label)
        segs = {}
        for code, act in ACTIVITY_CODE_MAP.items():
            m = codes == code
            if m.sum() < 2:
                continue
            x = _resample_uniform(ts[m] - ts[m][0], gsr[m], FS)
            xc, valid = E.clean_dropouts(x, FS, MAX_INTERP_SEC)
            if len(xc) < WS:                    # lebih pendek dari 1 window -> tak berguna
                continue
            xs = E.clean_nk(xc, FS) if use_nk else E.smooth(xc, FS, SMOOTH_CUTOFF)
            segs[act] = (xs, valid)
        if not segs:
            continue

        # 2) statistik z-score dari SELURUH aktivitas subjek (BEBAS LABEL)
        stats = E.subject_norm_stats(np.concatenate([v[0] for v in segs.values()]), BASELINE_NORM)

        # 3) window untuk SEMUA aktivitas; label diberikan pemanggil kemudian
        gidx = len(subjects); n0 = len(Xraw)
        for act, (xs, valid) in segs.items():
            xn = E.baseline_normalize(xs, BASELINE_NORM, stats)
            for i in range(0, len(xn) - WS + 1, HS):
                if not valid[i:i + WS].all():
                    continue
                win = xn[i:i + WS]
                if np.isnan(win).any():
                    continue
                Xraw.append(win.astype(np.float32)); groups.append(gidx)
                meta.append(f"{sid}|{act}")
        if len(Xraw) == n0:
            continue
        subjects.append(sid)
        if verbose:
            print(f"  {sid:14s}: {len(Xraw)-n0} window")

    return (np.asarray(Xraw, np.float32), np.asarray(groups, np.int64), meta, subjects)
