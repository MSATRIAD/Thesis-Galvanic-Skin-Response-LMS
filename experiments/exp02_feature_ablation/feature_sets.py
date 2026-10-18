"""
Dua himpunan fitur untuk eksperimen ablation:
  - BASIC    : 12 fitur (sama dgn shared.eda_preprocessing.eda_features)
  - EXTENDED : ~37 fitur EDA komprehensif (tonik, fasik, distribusi, dinamika,
               frekuensi, Hjorth)

Semua dihitung dari window yang SUDAH ternormalisasi baseline (z-score per subjek)
dan ter-smoothing — keluaran shared.data.build_raw_windows.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks

from ..shared import eda_preprocessing as E
from ..shared.config_shared import FS


def _safe(v):
    return float(v) if np.isfinite(v) else 0.0


# ---------------------------------------------------------------- BASIC (12)
BASIC_NAMES = ["tonic_mean", "tonic_std", "tonic_slope", "sig_mean", "sig_std",
               "sig_range", "phasic_std", "phasic_absmean", "scr_rate",
               "scr_amp_sum", "mean_abs_d1", "phasic_absmax"]


def basic(w, fs=FS):
    return E.eda_features(np.asarray(w, float), fs)


# ---------------------------------------------------------------- EXTENDED (~37)
EXTENDED_NAMES = [
    # tonik (6)
    "tonic_mean", "tonic_std", "tonic_slope", "tonic_min", "tonic_max", "tonic_range",
    # fasik (8)
    "phasic_std", "phasic_absmean", "scr_rate", "scr_amp_sum", "scr_count",
    "phasic_skew", "phasic_kurt", "phasic_area",
    # distribusi sinyal (11)
    "sig_mean", "sig_std", "sig_range", "sig_median", "sig_p25", "sig_p75",
    "sig_iqr", "sig_rms", "sig_mad", "sig_skew", "sig_kurt",
    # dinamika (5)
    "d1_absmean", "d1_absstd", "d1_absmax", "d2_absmean", "phasic_zcr",
    # frekuensi (4)
    "bp_0_0.1", "bp_0.1_0.2", "bp_0.2_0.4", "spectral_entropy",
    # Hjorth (3)
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
]


def extended(w, fs=FS):
    w = np.asarray(w, float)
    tonic, phasic = E.split_tonic_phasic(w, fs)
    rate, amp = E.scr_features(phasic, fs)
    sd = np.std(phasic)
    pk = (find_peaks(phasic, height=sd, distance=int(fs))[0]
          if sd > 1e-8 else np.array([], int))

    d1 = np.diff(w)
    d2 = np.diff(d1)
    zcr = (np.mean(np.abs(np.diff(np.sign(phasic)))) / 2.0) if len(phasic) > 1 else 0.0

    f, p = welch(w, fs=fs, nperseg=min(len(w), 128))
    psum = p.sum() + 1e-12
    band = lambda lo, hi: float(p[(f >= lo) & (f < hi)].sum())
    spec_ent = float(-np.sum((p / psum) * np.log(p / psum + 1e-12)))

    var0 = w.var(); var1 = d1.var() if len(d1) else 0.0; var2 = d2.var() if len(d2) else 0.0
    mob = np.sqrt(var1 / var0) if var0 > 0 else 0.0
    com = (np.sqrt(var2 / var1) / mob) if (var1 > 0 and mob > 0) else 0.0

    feats = [
        tonic.mean(), tonic.std(), np.polyfit(np.arange(len(tonic)), tonic, 1)[0],
        tonic.min(), tonic.max(), np.ptp(tonic),
        phasic.std(), np.abs(phasic).mean(), rate, amp, len(pk),
        skew(phasic), kurtosis(phasic), np.abs(phasic).sum() / fs,
        w.mean(), w.std(), np.ptp(w), np.median(w),
        np.percentile(w, 25), np.percentile(w, 75),
        np.percentile(w, 75) - np.percentile(w, 25),
        np.sqrt(np.mean(w ** 2)), np.mean(np.abs(w - w.mean())), skew(w), kurtosis(w),
        np.mean(np.abs(d1)), np.std(np.abs(d1)),
        np.max(np.abs(d1)) if len(d1) else 0.0, np.mean(np.abs(d2)) if len(d2) else 0.0, zcr,
        band(0, 0.1), band(0.1, 0.2), band(0.2, 0.4), spec_ent,
        var0, mob, com,
    ]
    return [_safe(x) for x in feats]
