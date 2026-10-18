"""
Representasi exp01: window GSR mentah -> heatmap GLCT (N,128,128).

Memperbaiki bug normalisasi: HEATMAP_NORM='global' memakai min-max log1p
GLOBAL (lintas semua window) sehingga perbedaan amplitudo antar-window
TIDAK terhapus (berbeda dgn 'per_image' yang menormalkan tiap gambar).
"""

import os
import hashlib
import numpy as np
from scipy.signal import butter, filtfilt
import cv2

from ..shared.config_shared import FS, WINDOW_SIZE
from ..shared.glct import glct_transform_fast
from . import config as C


def _signal_norm(Xraw, groups=None, subj_stats=None):
    if C.SIGNAL_NORM == "none":
        return Xraw
    if C.SIGNAL_NORM == "window":
        mu = Xraw.mean(1, keepdims=True); sd = Xraw.std(1, keepdims=True)
        return (Xraw - mu) / np.where(sd > 1e-8, sd, 1.0)
    if C.SIGNAL_NORM == "subject":
        assert groups is not None and subj_stats is not None
        mu = subj_stats[groups, 0:1]; sd = subj_stats[groups, 1:2]
        return (Xraw - mu) / np.where(sd > 1e-8, sd, 1.0)
    raise ValueError(C.SIGNAL_NORM)


def _highpass(x, fs=FS):
    nyq = 0.5 * fs
    wn = min(max(C.HPF_CUTOFF / nyq, 1e-3), 0.99)
    b, a = butter(C.HPF_ORDER, wn, btype="high")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - np.mean(x)
    return filtfilt(b, a, x)


def _tfr(x, fs=FS):
    chirps = np.linspace(C.GLCT_CHIRP_RANGE[0], C.GLCT_CHIRP_RANGE[1], C.GLCT_CHIRP_RATES)
    return glct_transform_fast(x, fs, n_freqs=C.GLCT_N_FREQS, chirp_rates=chirps,
                               window_length=C.GLCT_WINDOW_LENGTH,
                               hop_length=C.GLCT_HOP_LENGTH)


def _cache_key(n):
    s = f"{C.SIGNAL_NORM}|{C.HEATMAP_NORM}|{C.GLCT_N_FREQS}|{C.GLCT_WINDOW_LENGTH}|" \
        f"{C.GLCT_HOP_LENGTH}|{C.GLCT_CHIRP_RATES}|{C.GLCT_CHIRP_RANGE}|{n}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def build_heatmaps(Xraw, groups=None, subj_stats=None, force=False, verbose=True,
                   apply_hpf=True, tag=""):
    """
    apply_hpf=True  : ekstrak fasik via high-pass (window = sinyal ternormalisasi).
    apply_hpf=False : window SUDAH fasik (mis. dari cvxEDA) -> langsung GLCT.
    tag             : pembeda cache (mis. 'butter'/'cvxeda') agar tak bertabrakan.
    """
    cache = os.path.join(C.CACHE_DIR, f"heatmaps_{tag}_{int(apply_hpf)}_{_cache_key(len(Xraw))}.npy")
    if os.path.exists(cache) and not force:
        if verbose:
            print(f"[exp01 cache] {cache}")
        return np.load(cache)

    Xn = _signal_norm(Xraw.astype(np.float64), groups, subj_stats)

    tfr_logs = []
    for i in range(len(Xn)):
        sig = _highpass(Xn[i]) if apply_hpf else Xn[i]
        tfr = _tfr(sig)
        tfr_logs.append(np.log1p(tfr).astype(np.float32))
    # global min/max utk HEATMAP_NORM='global'
    gmin = min(t.min() for t in tfr_logs)
    gmax = max(t.max() for t in tfr_logs)

    H, W = C.CNN_INPUT_SIZE
    out = np.empty((len(Xn), H, W), dtype=np.float32)
    for i, lg in enumerate(tfr_logs):
        if C.HEATMAP_NORM == "global":
            nm = (lg - gmin) / (gmax - gmin + 1e-10)
        else:  # per_image
            mn, mx = lg.min(), lg.max()
            nm = np.zeros_like(lg) if (mx - mn) < 1e-10 else (lg - mn) / (mx - mn)
        out[i] = cv2.resize(nm, (W, H), interpolation=cv2.INTER_CUBIC)

    np.save(cache, out)
    if verbose:
        print(f"[exp01 cache] disimpan -> {cache}  shape={out.shape}")
    return out
