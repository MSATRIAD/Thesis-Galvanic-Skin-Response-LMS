"""
Pelabelan beban kognitif dari NASA-TLX WWL (bersama untuk semua arsitektur).

'extreme' : WWL <= P33 -> Low(0), >= P67 -> High(1), tengah dibuang.
'median'  : WWL < median -> Low(0), >= median -> High(1).
Label dihitung level (subjek x aktivitas), dipropagasi ke window.
Threshold dari subjek yang punya GSR sekaligus WWL.
"""

import re
import numpy as np
import pandas as pd

from .config_shared import (WWL_CSV, DETAIL_CSV, ACTIVITY_CODE_MAP, LABEL_STRATEGY,
                            EXTREME_LOW_Q, EXTREME_HIGH_Q)

ACTIVITY_NAMES = list(ACTIVITY_CODE_MAP.values())

# Sumber label: 'WWL' (komposit) atau dimensi NASA-TLX dari detail_all.csv.
# MD=Mental Demand (paling murni beban kognitif), EF=Effort, dst.
DIMENSION_COLS = {"MD": "Rating_MD", "PD": "Rating_PD", "TD": "Rating_TD",
                  "OP": "Rating_OP", "FR": "Rating_FR", "EF": "Rating_EF"}


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _filter_subjects(long, valid_subjects):
    long["subject_key"] = long["Respondent"].map(norm_name)
    if valid_subjects is not None:
        keys = {norm_name(s) for s in valid_subjects}
        long = long[long["subject_key"].isin(keys)].copy()
    return long


def load_wwl_long(valid_subjects=None):
    df = pd.read_csv(WWL_CSV)
    long = df.melt(id_vars="Respondent", value_vars=ACTIVITY_NAMES,
                   var_name="activity", value_name="wwl").dropna(subset=["wwl"])
    return _filter_subjects(long, valid_subjects)


def load_dimension_long(dim, valid_subjects=None):
    """Label dari satu dimensi NASA-TLX (mis. 'MD'). Kolom 'wwl' = rating dimensi."""
    col = DIMENSION_COLS[dim]
    df = pd.read_csv(DETAIL_CSV)
    long = df[["Respondent", "Activity", col]].rename(
        columns={"Activity": "activity", col: "wwl"}).dropna(subset=["wwl"])
    return _filter_subjects(long, valid_subjects)


def load_label_long(label_source="WWL", valid_subjects=None):
    if label_source == "WWL":
        return load_wwl_long(valid_subjects)
    return load_dimension_long(label_source, valid_subjects)


# Kategori workload NASA-TLX absolut (5 kelas), berdasarkan rentang WWL 0-100.
WORKLOAD5 = [(9, "Low"), (29, "Medium"), (49, "Somewhat High"),
             (79, "High"), (100, "Very High")]


def _workload5(wwl):
    for cls, (hi, _name) in enumerate(WORKLOAD5):
        if wwl <= hi:
            return cls
    return len(WORKLOAD5) - 1


def build_label_table(valid_subjects=None, strategy=None, label_source="WWL"):
    """
    strategy: 'extreme' (biner tertil) | 'median' (biner) | 'workload5' (5 kelas absolut).
    label_source: 'WWL' (komposit) | 'MD'/'EF'/dst (dimensi NASA-TLX, detail_all.csv).
    Default None -> pakai LABEL_STRATEGY dari config.
    """
    strategy = strategy or LABEL_STRATEGY
    long = load_label_long(label_source, valid_subjects)
    w = long["wwl"].values
    info = {"strategy": strategy, "n": len(w),
            "median": float(np.median(w)),
            "p33": float(np.quantile(w, EXTREME_LOW_Q)),
            "p67": float(np.quantile(w, EXTREME_HIGH_Q)),
            "mean": float(np.mean(w)), "std": float(np.std(w))}

    labels, counts, n_drop = {}, {}, 0
    for _, row in long.iterrows():
        wwl = row["wwl"]
        if strategy == "extreme":
            lab = 0 if wwl <= info["p33"] else 1 if wwl >= info["p67"] else None
        elif strategy == "median":
            lab = 0 if wwl < info["median"] else 1
        elif strategy == "workload5":
            lab = _workload5(wwl)
        elif strategy == "cl3":
            # ambang tetap: Rendah 0-33 (0) | Sedang 34-66 (1) | Tinggi 67-100 (2)
            lab = 0 if wwl <= 33 else 1 if wwl <= 66 else 2
        else:
            raise ValueError(strategy)
        if lab is None:
            n_drop += 1; continue
        labels[(row["subject_key"], row["activity"])] = lab
        counts[lab] = counts.get(lab, 0) + 1

    info["class_counts"] = dict(sorted(counts.items()))
    info["n_dropped"] = n_drop
    if strategy == "workload5":
        info["class_names"] = [n for _, n in WORKLOAD5]
    elif strategy == "cl3":
        info["class_names"] = ["Rendah", "Sedang", "Tinggi"]
    return labels, info


def get_label(labels, subject_name, activity_name):
    return labels.get((norm_name(subject_name), activity_name), None)
