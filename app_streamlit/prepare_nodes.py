"""Ekspor 54 node latih (fitur EXTENDED per aktivitas + label) ke nodes_train.npz,
beserta ambang tertil dan tabel WWL sebagai KEBENARAN DASAR untuk menghitung akurasi.

Berkas inilah yang HARUS dibawa bersama aplikasi, sebab GCN bersifat transduktif:
tidak ada berkas bobot model yang bisa dipakai ulang, graf harus dibangun ulang
setiap ada subjek baru. Jalankan sekali:  python -m app_streamlit.prepare_nodes
"""
import os

import numpy as np
import pandas as pd

from experiments.shared.config_shared import WWL_CSV, ACTIVITY_CODE_MAP
from experiments.shared.data import build_raw_windows, list_files, subject_from_path
from experiments.shared.labels_nasatlx import build_label_table, norm_name
from experiments.exp02_feature_ablation.feature_sets import EXTENDED_NAMES
from experiments.exp35_microsiemens_ab.run import activity_nodes

OUT = os.path.join(os.path.dirname(__file__), "nodes_train.npz")


def main():
    X, y, g, subjects, meta, _ = build_raw_windows(
        verbose=False, label_strategy="extreme",
        window_sec=60, hop_sec=30, use_nk=True)
    Xn, yn, gn = activity_nodes(X, y, g, meta)

    seen, names = set(), []
    for m in meta:
        if m not in seen:
            seen.add(m); names.append(m)
    assert len(names) == len(Xn)

    # Ambang tertil global (P33/P67) HARUS dihitung dari himpunan subjek yang sama
    # dengan pipeline (22 responden utama dari DATA_DIR, TANPA 5 subjek pilot),
    # bukan dari seluruh baris summary_wwl.csv.
    wwl = pd.read_csv(WWL_CSV)
    acts = list(ACTIVITY_CODE_MAP.values())
    pipeline_subjects = sorted({subject_from_path(f) for f in list_files()
                                if not os.path.basename(f).startswith("_")})
    _, info = build_label_table(valid_subjects=pipeline_subjects, strategy="extreme")
    p33, p67 = float(info["p33"]), float(info["p67"])

    # tabel WWL (kebenaran dasar) untuk semua responden yang punya NASA-TLX
    rows = [(norm_name(r["Respondent"]), a, float(r[a]))
            for _, r in wwl.iterrows() for a in acts if pd.notna(r[a])]

    np.savez_compressed(
        OUT, Xn=Xn.astype(np.float32), yn=yn.astype(np.int64), gn=gn.astype(np.int64),
        node_names=np.array(names, dtype=object),
        subjects=np.array(subjects, dtype=object),
        feature_names=np.array(EXTENDED_NAMES, dtype=object),
        p33=np.float64(p33), p67=np.float64(p67),
        wwl_keys=np.array([f"{s}|{a}" for s, a, _ in rows], dtype=object),
        wwl_vals=np.array([v for _, _, v in rows], dtype=np.float64))

    print(f"nodes_train.npz -> {OUT}")
    print(f"  node        : {Xn.shape[0]} aktivitas x {Xn.shape[1]} fitur")
    print(f"  label       : Rendah={int((yn == 0).sum())}  Tinggi={int((yn == 1).sum())}"
          f"  (base rate {max(np.bincount(yn)) / len(yn):.4f})")
    print(f"  subjek latih: {len(subjects)}")
    print(f"  ambang EGA  : P33={p33:.2f}  P67={p67:.2f}  (di antaranya = tertil tengah)")
    print(f"  WWL rujukan : {len(rows)} aktivitas dari {wwl.shape[0]} responden")


if __name__ == "__main__":
    main()
