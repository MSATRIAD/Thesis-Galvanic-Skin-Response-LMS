"""Siapkan data latih untuk KEEMPAT model -> windows_train.npz.

Berisi window mentah (untuk AttnBiLSTM), heatmap GLCT (untuk CNN), node aktivitas
(untuk SVM & GCN), label, dan nama node. Semua model dilatih ULANG saat aplikasi
dipakai, dengan subjek yang diuji DIKELUARKAN lebih dulu, sehingga akurasi yang
terukur selalu jujur (lintas-subjek).

Jalankan sekali:  python -m app_streamlit.prepare_models
"""
import os

import numpy as np

from experiments.shared.data import build_raw_windows
from experiments.exp01_glct_cnn.representation import build_heatmaps
from experiments.exp02_feature_ablation.feature_sets import EXTENDED_NAMES
from experiments.exp35_microsiemens_ab.run import activity_nodes

OUT = os.path.join(os.path.dirname(__file__), "windows_train.npz")


def main():
    X, y, g, subjects, meta, st = build_raw_windows(
        verbose=False, label_strategy="extreme",
        window_sec=60, hop_sec=30, use_nk=True)
    print(f"window: {X.shape} | subjek: {len(subjects)}")

    H = build_heatmaps(X, groups=g, subj_stats=st, apply_hpf=True,
                       tag="app60", verbose=False)
    print(f"heatmap GLCT: {H.shape}")

    Xn, yn, gn = activity_nodes(X, y, g, meta)
    seen, names = set(), []
    for m in meta:
        if m not in seen:
            seen.add(m); names.append(m)
    print(f"node aktivitas: {Xn.shape}")

    np.savez_compressed(
        OUT,
        Xraw=X.astype(np.float32), H=H.astype(np.float32),
        y=y.astype(np.int64), meta=np.array(meta, dtype=object),
        Xn=Xn.astype(np.float32), yn=yn.astype(np.int64),
        node_names=np.array(names, dtype=object),
        subjects=np.array(subjects, dtype=object),
        feature_names=np.array(EXTENDED_NAMES, dtype=object))
    mb = os.path.getsize(OUT) / 1e6
    print(f"-> {OUT}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
