"""
Preprocessing EDA/GSR yang benar (bersama untuk semua eksperimen).

Tiga perbaikan utama:
  1. DROPOUT: interpolasi gap PENDEK (<= max_interp_sec), TANDAI gap PANJANG
     sebagai invalid (window yang menyentuhnya dibuang — bukan diisi garis palsu).
  2. SMOOTHING + DEKOMPOSISI: low-pass ~1 Hz buang derau; pisah tonik (SCL) & fasik (SCR).
  3. PENYAMAAN BASELINE per-subjek: 'range' (Lykken) atau 'zscore'.

Catatan: gsrRaw dalam unit ADC sembarang (bukan µS), jadi ambang SCR memakai
kelipatan std (relatif), bukan ambang absolut literatur.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def cvxeda(y, fs, tau0=2.0, tau1=0.7, delta_knot=10.0, alpha=8e-4, gamma=1e-2,
           solver=None, reltol=1e-8):
    """
    Dekomposisi EDA via optimasi konveks (cvxEDA, Greco et al. 2016, IEEE TBME).
    Memodelkan SC = fasik (r) + tonik (t) + noise. Fasik = aktivitas saraf
    sudomotor sparse dikonvolusi respons Bateman. Implementasi resmi (BSD,
    lciti/cvxEDA). Input sebaiknya ter-standardisasi (z-score).

    Returns: r (fasik/SCR), t (tonik/SCL), p (driver sparse), e (noise).
    """
    import cvxopt as cv
    import cvxopt.solvers
    delta = 1.0 / fs
    n = len(y)
    y = cv.matrix(np.asarray(y, dtype=float))

    a1 = 1.0 / min(tau1, tau0); a0 = 1.0 / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta ** 2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta ** 2)
    ma = np.array([1., 2., 1.])
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))
    M = cv.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))

    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0., -1.)]
    spl = np.convolve(spl, spl, 'full'); spl /= max(spl)
    i = np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n + 1.) / n])
    nC = C.size[1]

    old = cv.solvers.options.copy(); cv.solvers.options.clear()
    cv.solvers.options.update({'reltol': reltol, 'show_progress': False})
    Mt, Ct, Bt = M.T, C.T, B.T
    H = cv.sparse([[Mt * M, Ct * M, Bt * M], [Mt * C, Ct * C, Bt * C],
                   [Mt * B, Ct * B, Bt * B + gamma * cv.spmatrix(1.0, range(nB), range(nB))]])
    f = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
    z = lambda a, b: cv.spmatrix([], [], [], (a, b))   # konstrain A*q>=0 pd variabel penuh
    G = cv.sparse([[cv.spmatrix(-A.V, A.I, A.J, (n, n))], [z(n, nC)], [z(n, nB)]])
    res = cv.solvers.qp(H, f, G, cv.matrix(0., (n, 1)), solver=solver)
    cv.solvers.options.clear(); cv.solvers.options.update(old)

    l = res['x'][-nB:]; d = res['x'][n:n + nC]
    t = B * l + C * d
    q = res['x'][:n]; r = M * q; e = y - r - t
    return (np.array(r).ravel(), np.array(t).ravel(),
            np.array(A * q).ravel(), np.array(e).ravel())


def _butter(x, fs, cutoff, btype, order=2):
    wn = float(np.clip(cutoff / (fs / 2.0), 1e-4, 0.99))
    b, a = butter(order, wn, btype=btype)
    if len(x) <= 3 * max(len(a), len(b)):
        return (x - np.mean(x)) if btype == "high" else x
    return filtfilt(b, a, x)


def clean_dropouts(raw, fs, max_interp_sec=2.0):
    """
    Kembalikan (x_bersih, valid_mask).
    Gap 0 pendek diinterpolasi; gap panjang tetap diisi tapi valid_mask=False
    sehingga window yang menyentuhnya bisa dibuang di tahap segmentasi.
    """
    x = raw.astype(float).copy()
    valid = np.ones(len(x), dtype=bool)
    zero = x == 0
    if zero.any():
        idx = np.where(zero)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in runs:
            if len(run) > max_interp_sec * fs:
                valid[run] = False
        x[zero] = np.nan
        x = pd.Series(x).interpolate("linear", limit_direction="both").values
        if np.isnan(x).any():                       # bila seluruhnya NaN
            x = np.nan_to_num(x, nan=np.nanmean(raw[raw > 0]) if (raw > 0).any() else 0.0)
    return x, valid


def smooth(x, fs, cutoff=1.0):
    """Low-pass ~1 Hz: buang derau di atas pita EDA."""
    return _butter(x, fs, cutoff, "low")


def clean_nk(x, fs):
    """Pembersihan EDA via NeuroKit2 (low-pass Butterworth, metode default 'neurokit';
    Makowski dkk., 2021). Disediakan agar praproses dapat disitasi ke pustaka baku."""
    import neurokit2 as nk
    return np.asarray(nk.eda_clean(np.asarray(x, dtype=float), sampling_rate=fs), dtype=float)


def split_tonic_phasic(x, fs, tonic_cutoff=0.05):
    """tonic = komponen sangat lambat (SCL); phasic = sisanya (SCR)."""
    tonic = _butter(x, fs, tonic_cutoff, "low")
    return tonic, x - tonic


def baseline_normalize(x, method, stats=None):
    """
    Samakan baseline per-subjek.
      'none'   : apa adanya
      'range'  : (x - min) / (max - min)            [Lykken range-correction]
      'zscore' : (x - mean) / std
    `stats` = (a, b) statistik level-SUBJEK (min,max) atau (mean,std); bila None
    dihitung dari x itu sendiri.
    """
    if method == "none":
        return x
    if method == "range":
        mn, mx = stats if stats else (np.nanmin(x), np.nanmax(x))
        return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)
    if method == "zscore":
        mu, sd = stats if stats else (np.nanmean(x), np.nanstd(x))
        return (x - mu) / sd if sd > 1e-8 else x - mu
    raise ValueError(method)


def subject_norm_stats(signal_concat, method):
    """Hitung statistik normalisasi level-subjek dari gabungan sinyal task subjek."""
    if method == "range":
        return (float(np.nanmin(signal_concat)), float(np.nanmax(signal_concat)))
    if method == "zscore":
        return (float(np.nanmean(signal_concat)), float(np.nanstd(signal_concat)))
    return None


def scr_features(phasic, fs, thr_std=1.0, min_dist_sec=1.0):
    """Fitur fasik proper: laju SCR/menit + total amplitudo."""
    sd = np.nanstd(phasic)
    if sd <= 1e-8:
        return 0.0, 0.0
    pk, props = find_peaks(phasic, height=sd * thr_std, distance=int(fs * min_dist_sec))
    rate = len(pk) / (len(phasic) / fs) * 60.0
    amp = float(props["peak_heights"].sum()) if len(pk) else 0.0
    return rate, amp


def eda_features(window_norm, fs):
    """12 fitur dari window yang SUDAH dinormalisasi baseline (tonik + fasik bersih)."""
    tonic, phasic = split_tonic_phasic(window_norm, fs)
    rate, amp = scr_features(phasic, fs)
    return [
        np.mean(tonic), np.std(tonic), np.polyfit(np.arange(len(tonic)), tonic, 1)[0],
        np.mean(window_norm), np.std(window_norm), np.ptp(window_norm),
        np.std(phasic), np.mean(np.abs(phasic)), rate, amp,
        np.mean(np.abs(np.diff(window_norm))), np.max(np.abs(phasic)),
    ]
