"""
Pemuatan data BERSAMA (representation-agnostic).

Menghasilkan window sinyal GSR MENTAH (sudah dibersihkan + resample uniform,
TANPA normalisasi / filter) beserta label NASA-TLX dan grup subjek.
Tiap arsitektur lalu mengubah window mentah ini menjadi representasinya
sendiri (GLCT, STFT, raw 1D, dll) -> perbandingan adil karena window identik.

Output build_raw_windows():
    Xraw     : (N, WINDOW_SIZE) float32  sinyal GSR mentah per window
    y        : (N,) int64                label 0/1
    groups   : (N,) int64                indeks subjek
    subjects : list[str]                 nama subjek per indeks grup
    meta     : list[str]                 "subjek|aktivitas" per window
    subj_stats : (n_subjek, 2) float32   (mean, std) sinyal task tiap subjek
                                         (utk opsi normalisasi per-subjek)
"""

import os
import re
import glob
import numpy as np
import pandas as pd

from .config_shared import (DATA_DIR, SHARED_CACHE, FS, GSR_COLUMN, TIME_COLUMN,
                           ACTIVITY_CODE_COLUMN, ACTIVITY_CODE_MAP, WINDOW_SIZE,
                           HOP_SIZE, MAX_INTERP_SEC, SMOOTH_CUTOFF, BASELINE_NORM,
                           PROJECT_ROOT)
from .labels_nasatlx import build_label_table, get_label
from . import eda_preprocessing as E


# Berkas mentah pilot tidak dibundel dalam repo publik. Cache evaluasi tetap
# menyimpan metadata pilot dengan ID anonim `pilot_N`.
PILOT_FILES = []


def subject_from_path(path):
    b = os.path.basename(path)
    m = re.match(r"(.+?)_\d{8}_", b)
    return m.group(1) if m else b.replace(".csv", "")


def list_files(data_dir=None):
    """Daftar CSV rekaman sensor, tanpa CSV turunan/metadata di root repo.

    Dataset utama disimpan dalam folder ``data pc N``. Untuk eksperimen yang
    secara eksplisit memberikan ``data_dir`` lain, pertahankan pencarian
    rekursif seperti implementasi awal.
    """
    if data_dir is None:
        pattern = os.path.join(DATA_DIR, "data pc *", "*.csv")
        return sorted(glob.glob(pattern))
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))


def _clean_dropouts(x):
    if not (x == 0.0).any():
        return x
    s = pd.Series(x).replace(0.0, np.nan).interpolate("linear").ffill().bfill()
    return s.fillna(s.mean() if not s.isna().all() else 0.0).values


def _resample_uniform(t_sec, x, fs=FS):
    if len(t_sec) < 2:
        return x
    n = max(2, int((t_sec[-1] - t_sec[0]) * fs) + 1)
    return np.interp(np.linspace(t_sec[0], t_sec[-1], n), t_sec, x)


def _rest_stats(df, gsr, method, verbose_name=""):
    """Statistik baseline dari segmen ISTIRAHAT murni (LMS_Activity=='resting').
    Mengembalikan (a,b) ala subject_norm_stats, atau None bila rest tak memadai."""
    if "LMS_Activity" not in df.columns:
        return None
    rmask = (df["LMS_Activity"].astype(str) == "resting").values
    r = gsr[rmask]
    r = r[r > 0]                       # buang dropout (0)
    if len(r) < FS * 5:               # butuh >=5 detik rest bersih
        return None
    return E.subject_norm_stats(r.astype(float), method)


def build_raw_windows(force=False, verbose=True, label_strategy=None,
                      include_pilot=False, baseline_ref="task", label_source="WWL",
                      data_dir=None, gsr_column=None, data_tag="",
                      window_sec=None, hop_sec=None, use_nk=True):
    """label_strategy: None -> pakai config LABEL_STRATEGY; atau 'extreme'/'median'/'workload5'.
    include_pilot: bila True, tambahkan 5 subjek Data_Pilot (cache & hasil terpisah).
    baseline_ref: 'task' (default, mean&std dari sinyal task)
                | 'rest'     (mean&std dari ISTIRAHAT -> (x-mean_rest)/std_rest)
                | 'restmean' (pusat rest, skala task -> (x-mean_rest)/std_task;
                              isolasi 'offset dari rest' tanpa std_rest tak stabil).
    label_source: 'WWL' (komposit) | 'MD'/'EF'/dst (dimensi NASA-TLX).
    Cache dipisah per strategi (pilot/baseline_ref/label_source) agar tak saling menimpa."""
    from .config_shared import LABEL_STRATEGY
    strat = label_strategy or LABEL_STRATEGY
    # panjang/hop window: default config (30s/15s); bisa di-override (mis. sweep exp37)
    _WS = int(FS * window_sec) if window_sec else WINDOW_SIZE
    _HS = int(FS * hop_sec) if hop_sec else HOP_SIZE
    _win_sfx = f"_w{window_sec}h{hop_sec}" if window_sec else ""
    _win_sfx += "_nk" if use_nk else ""
    _ref_sfx = {"task": "", "rest": "_restbase", "restmean": "_restmean"}[baseline_ref]
    _src_sfx = "" if label_source == "WWL" else f"_{label_source}"
    suffix = ("_pilot" if include_pilot else "") + _ref_sfx + _src_sfx + _win_sfx
    os.makedirs(SHARED_CACHE, exist_ok=True)
    cache = os.path.join(SHARED_CACHE, f"clean_windows_{data_tag}{BASELINE_NORM}_{strat}{suffix}.npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[shared cache] {cache} -> Xraw{d['Xraw'].shape}")
        return (d["Xraw"], d["y"], d["groups"], list(d["subjects"]),
                list(d["meta"]), d["subj_stats"])

    # pasangan (file, nama_subjek): folder utama via regex, pilot via nama eksplisit
    _files = [f for f in list_files(data_dir)
              if not os.path.basename(f).startswith("_")]   # buang berkas laporan _*.csv
    pairs = [(f, subject_from_path(f)) for f in _files]
    if include_pilot:
        pairs += [(p, name) for name, p in PILOT_FILES]
    labels, info = build_label_table(valid_subjects=[sid for _, sid in pairs],
                                     strategy=strat, label_source=label_source)
    if verbose:
        print("=== Pelabelan NASA-TLX (shared) ===")
        for k, v in info.items():
            print(f"  {k:18s}: {v}")
        print("=" * 60)

    Xraw, y, groups, meta, subjects, subj_stats = [], [], [], [], [], []
    for f, sid in pairs:
        if not any(get_label(labels, sid, a) is not None for a in ACTIVITY_CODE_MAP.values()):
            if verbose:
                print(f"  [SKIP] {sid}: tak ada label WWL")
            continue

        df = pd.read_csv(f)
        _ts = pd.to_datetime(df[TIME_COLUMN])
        ts = (_ts - _ts.iloc[0]).dt.total_seconds().values
        codes = df[ACTIVITY_CODE_COLUMN].values
        gsr = df[gsr_column or GSR_COLUMN].values.astype(float)

        gidx = len(subjects)
        n0 = len(Xraw)

        # 1) Per aktivitas: resample -> bersihkan dropout (interp pendek, tandai gap panjang)
        #    -> smoothing low-pass 1 Hz. Simpan (sinyal_bersih, valid_mask).
        segs = {}
        for code, act in ACTIVITY_CODE_MAP.items():
            lab = get_label(labels, sid, act)
            if lab is None:
                continue
            m = codes == code
            if m.sum() < 2:
                continue
            x = _resample_uniform(ts[m] - ts[m][0], gsr[m], FS)
            xc, valid = E.clean_dropouts(x, FS, MAX_INTERP_SEC)
            xs = E.clean_nk(xc, FS) if use_nk else E.smooth(xc, FS, SMOOTH_CUTOFF)
            segs[act] = (xs, valid, lab)

        if not segs:
            continue

        # 2) Statistik baseline level-SUBJEK.
        #    baseline_ref='task': dari gabungan sinyal task bersih (default).
        #    baseline_ref='rest': dari segmen ISTIRAHAT (change-from-rest);
        #                         fallback ke task bila rest tak memadai.
        concat = np.concatenate([v[0] for v in segs.values()])
        stats = E.subject_norm_stats(concat, BASELINE_NORM)   # (mean_task, std_task)
        if baseline_ref in ("rest", "restmean"):
            rstats = _rest_stats(df, gsr, BASELINE_NORM)
            if rstats is not None:
                # 'rest'    : (mean_rest, std_rest)
                # 'restmean': (mean_rest, std_task)  -> pusat rest, skala task
                stats = rstats if baseline_ref == "rest" else (rstats[0], stats[1])
            elif verbose:
                print(f"  [{baseline_ref}->task fallback] {sid}: rest tak memadai")

        # 3) Normalisasi baseline per-subjek -> window (buang window dgn dropout panjang)
        for act, (xs, valid, lab) in segs.items():
            xn = E.baseline_normalize(xs, BASELINE_NORM, stats)
            for i in range(0, len(xn) - _WS + 1, _HS):
                if not valid[i:i + _WS].all():             # window menyentuh dropout panjang
                    continue
                win = xn[i:i + _WS]
                if np.isnan(win).any():
                    continue
                Xraw.append(win.astype(np.float32))
                y.append(lab); groups.append(gidx); meta.append(f"{sid}|{act}")

        if len(Xraw) == n0:        # subjek tak menghasilkan window (mis. file rusak)
            continue
        subjects.append(sid)
        subj_stats.append(list(stats) if stats else [0.0, 1.0])
        if verbose:
            print(f"  {sid:14s} (subj#{gidx:2d}): {len(Xraw)-n0} windows")

    Xraw = np.asarray(Xraw, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    subj_stats = np.asarray(subj_stats, dtype=np.float32)

    if verbose:
        print("=" * 60)
        print(f"TOTAL: {len(Xraw)} windows | {len(subjects)} subjek "
              f"| Low={int((y==0).sum())} High={int((y==1).sum())}")

    np.savez_compressed(cache, Xraw=Xraw, y=y, groups=groups,
                        subjects=np.array(subjects, dtype=object),
                        meta=np.array(meta, dtype=object), subj_stats=subj_stats)
    if verbose:
        print(f"[shared cache] disimpan -> {cache}")
    return Xraw, y, groups, subjects, meta, subj_stats
