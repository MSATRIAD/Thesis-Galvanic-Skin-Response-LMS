"""Inti inferensi aplikasi: praproses CSV baru -> node aktivitas -> prediksi.

Dua jalur, sengaja dipisah supaya bedanya kelihatan:
  - SVM  (INDUKTIF)   : fit sekali pada 54 node latih, lalu predict node baru.
  - GCN  (TRANSDUKTIF): graf k-NN dibangun ULANG atas (54 node latih + node baru),
                        dilatih ulang dengan loss di-mask ke node berlabel saja.
                        Tidak ada bobot model yang bisa dipakai ulang.

GCN dijalankan sebagai ENSEMBLE 10 seed (suara terbanyak), karena satu seed tunggal
terbukti rapuh: sekitar sepertiga prediksi berubah bila seed diganti. Jumlah suara
sekaligus menjadi ukuran keyakinan yang jujur (mengukur kestabilan keputusan).
"""
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.shared import eda_preprocessing as E
from experiments.shared.config_shared import (
    DEVICE, FS, GSR_COLUMN, TIME_COLUMN, ACTIVITY_CODE_COLUMN, ACTIVITY_CODE_MAP,
    MAX_INTERP_SEC, BASELINE_NORM)
from experiments.shared.data import _resample_uniform
from experiments.shared.evaluation import _class_weights
from experiments.exp01_glct_cnn import config as C1
from experiments.exp01_glct_cnn.model import CognitiveLoadCNN
from experiments.exp01_glct_cnn.representation import build_heatmaps
from experiments.exp02_feature_ablation.feature_sets import extended
from experiments.exp35_microsiemens_ab.run import GCN, knn_adj, norm_adj, pipe
from experiments.exp40_master60_honest.run import AttnBiLSTM

WINDOW_SEC, HOP_SEC = 60, 30
WS, HS = FS * WINDOW_SEC, FS * HOP_SEC
SEEDS = [0, 1, 7, 13, 42, 99, 123, 777, 2024, 31337]
KELAS = ["Rendah", "Tinggi"]
REQUIRED_COLS = [TIME_COLUMN, GSR_COLUMN, ACTIVITY_CODE_COLUMN]


# ------------------------------------------------------------------ praproses
def validate(df):
    """Kembalikan daftar masalah pada CSV (kosong = lolos)."""
    problems = []
    for c in REQUIRED_COLS:
        if c not in df.columns:
            problems.append(f"kolom wajib '{c}' tidak ada")
    if problems:
        return problems
    if not set(df[ACTIVITY_CODE_COLUMN].unique()) & set(ACTIVITY_CODE_MAP):
        problems.append("tidak ada Activity_Code 1-4 (watching/reading/discussion/quiz)")
    return problems


def preprocess(df, baseline_ref="task"):
    """CSV mentah satu subjek -> dict per aktivitas berisi sinyal, window, fitur.

    baseline_ref='task' : statistik z-score dari seluruh sinyal tugas (butuh sesi utuh).
    baseline_ref='rest' : statistik dari segmen istirahat saja (memungkinkan inferensi
                          daring, tanpa menunggu sesi selesai).
    """
    ts = pd.to_datetime(df[TIME_COLUMN])
    ts = (ts - ts.iloc[0]).dt.total_seconds().values
    codes = df[ACTIVITY_CODE_COLUMN].values
    gsr = df[GSR_COLUMN].values.astype(float)

    segs = {}
    for code, act in ACTIVITY_CODE_MAP.items():
        m = codes == code
        if m.sum() < 2:
            continue
        x = _resample_uniform(ts[m] - ts[m][0], gsr[m], FS)
        xc, valid = E.clean_dropouts(x, FS, MAX_INTERP_SEC)
        segs[act] = (E.clean_nk(xc, FS), valid)
    if not segs:
        return {}, None

    # statistik normalisasi per-subjek
    stats = E.subject_norm_stats(
        np.concatenate([v[0] for v in segs.values()]), BASELINE_NORM)
    used_rest = False
    if baseline_ref == "rest" and "LMS_Activity" in df.columns:
        r = gsr[(df["LMS_Activity"].astype(str) == "resting").values]
        r = r[r > 0]
        if len(r) >= FS * 5:
            stats = E.subject_norm_stats(r.astype(float), BASELINE_NORM)
            used_rest = True

    out = {}
    for act, (xs, valid) in segs.items():
        xn = E.baseline_normalize(xs, BASELINE_NORM, stats)
        wins, n_reject = [], 0
        for i in range(0, len(xn) - WS + 1, HS):
            w = xn[i:i + WS]
            if not valid[i:i + WS].all() or np.isnan(w).any():
                n_reject += 1
                continue
            wins.append(w.astype(np.float32))
        if not wins:
            out[act] = dict(signal=xn, windows=[], feats=None, n_reject=n_reject,
                            durasi=len(xn) / FS)
            continue
        F = np.nan_to_num(np.array([extended(w, FS) for w in wins]))
        out[act] = dict(signal=xn, windows=wins, feats=F, node=F.mean(0),
                        n_reject=n_reject, durasi=len(xn) / FS)
    return out, dict(stats=stats, used_rest=used_rest)


# ------------------------------------------------------------------ prediksi
def predict_svm(Xn, yn, Xnew):
    """INDUKTIF: satu model, dipakai ulang. Data latih tak perlu ikut saat inferensi."""
    m = clone(pipe(SVC(C=1, kernel="rbf", class_weight="balanced",
                       probability=True, random_state=42)))
    m.fit(Xn, yn)
    return m.predict(Xnew), m.predict_proba(Xnew)[:, 1]


def _gcn_once(Xn, yn, Xnew, seed):
    Xall = np.vstack([Xn, Xnew])
    Xs = StandardScaler().fit_transform(Xall)
    A = norm_adj(knn_adj(Xs, 8)).astype(np.float32)
    At, Xt = torch.tensor(A), torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(np.concatenate([yn, np.zeros(len(Xnew), int)]), dtype=torch.long)
    tm = torch.zeros(len(Xall), dtype=torch.bool)
    tm[:len(Xn)] = True                       # HANYA node latih yang berlabel

    torch.manual_seed(seed); np.random.seed(seed)
    m = GCN(Xs.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss(weight=_class_weights(yn, 2))
    for _ in range(150):
        m.train(); opt.zero_grad()
        crit(m(At, Xt)[tm], yt[tm]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(At, Xt).argmax(1).numpy()[len(Xn):]


def predict_gcn(Xn, yn, Xnew, seeds=SEEDS):
    """TRANSDUKTIF: graf dibangun ulang, model dilatih ulang, tiap seed sekali.
    Kembali: (label mayoritas, jumlah seed yang memilih 'Tinggi')."""
    P = np.array([_gcn_once(Xn, yn, Xnew, s) for s in seeds])   # (n_seed, n_new)
    votes = P.sum(0)
    return (votes >= len(seeds) / 2).astype(int), votes


def subject_from_filename(name):
    """'r_5_20260609_152917.csv' -> 'r_5' (aturan sama dgn shared/data.py)."""
    m = re.match(r"(.+?)_\d{8}_", os.path.basename(name))
    return m.group(1) if m else os.path.basename(name).replace(".csv", "")


def resolve_wwl_subject(fname, wwl_keys):
    """Cocokkan nama berkas dengan nama responden di tabel NASA-TLX.

    Dataset utama memakai ID anonim ``r_N``. Berkas pilot lama dapat masih memakai
    awalan ``R00N``; awalan tersebut dipetakan ke ID anonim ``pilot_N`` tanpa
    memerlukan nama asli. Kembalikan kunci ternormalisasi, atau None.
    """
    raw = norm(subject_from_filename(fname))
    kandidat = {norm(k.split("|")[0]) for k in wwl_keys}
    if raw in kandidat:
        return raw
    pilot = re.match(r"r00([1-5])", raw)
    if pilot and f"pilot{pilot.group(1)}" in kandidat:
        return f"pilot{pilot.group(1)}"
    # cocokkan sebagai bagian dari nama berkas (mis. 'r002evan' memuat 'evan'),
    # ambil yang paling panjang supaya tak salah pilih nama pendek yang kebetulan cocok
    cocok = [c for c in kandidat if c and c in raw]
    return max(cocok, key=len) if cocok else None


def exclude_subject(Xn, yn, node_names, subject):
    """Buang seluruh node milik `subject` dari data latih.

    WAJIB dipakai bila subjek yang diuji ternyata ADA di data latih. Tanpa ini,
    model memprediksi orang yang sudah pernah ia lihat, dan akurasinya palsu
    (bukan lagi subject-independent).
    """
    key = norm(subject)
    keep = [i for i, n in enumerate(node_names) if norm(n.split("|")[0]) != key]
    return Xn[keep], yn[keep], [node_names[i] for i in keep], len(Xn) - len(keep)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def true_label(wwl, p33, p67):
    """Label sebenarnya menurut Extreme Groups Approach.
    Mengembalikan 0 (Rendah), 1 (Tinggi), atau None bila jatuh di TERTIL TENGAH
    (dibuang saat pelatihan -> tak ada kebenaran dasar, akurasi tak bisa dihitung)."""
    if wwl is None:
        return None
    if wwl <= p33:
        return 0
    if wwl >= p67:
        return 1
    return None


def neighbors(Xn, Xnew, k=8):
    """Tetangga k-NN tiap node baru pada graf gabungan -> untuk visualisasi.
    Mengembalikan indeks yang menunjuk ke Xn (node latih) atau ke node baru lain."""
    Xs = StandardScaler().fit_transform(np.vstack([Xn, Xnew]))
    A = knn_adj(Xs, k)
    n = len(Xn)
    return [np.nonzero(A[n + i])[0] for i in range(len(Xnew))]


# ============================================================ MODEL LEVEL-WINDOW
# GLCT-CNN dan AttnBiLSTM bekerja per WINDOW 60 detik, bukan per aktivitas. Supaya
# keempat model dinilai pada SATUAN YANG SAMA (aktivitas), prediksi window digabung
# lewat AGREGASI: probabilitas dirata-ratakan per aktivitas, lalu diambil kelas
# tertinggi. Ini sekaligus menghapus bias durasi (aktivitas panjang punya lebih
# banyak window, sehingga tanpa agregasi ia berbobot lebih besar).

def _train_torch(model, Xtr, ytr, epochs=15, bs=32, lr=1e-3, wd=1e-4):
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                  torch.from_numpy(ytr).long()), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=_class_weights(ytr, 2).to(DEVICE))
    model.to(DEVICE)
    for _ in range(epochs):                      # epoch TETAP: tak ada early stopping,
        model.train()                            # sehingga lipatan uji tak pernah diintip
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
    model.eval()
    return model


def _proba(model, X, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(DEVICE)
            out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.vstack(out)


def _aggregate(proba, act_of_window, acts):
    """Rata-ratakan probabilitas window -> satu keputusan per aktivitas."""
    lab, conf = [], []
    for a in acts:
        m = np.array(act_of_window) == a
        p = proba[m].mean(0)
        lab.append(int(p.argmax())); conf.append(float(p[1]))
    return np.array(lab), np.array(conf)


def predict_cnn(Htr, ytr, Hnew, act_of_window, acts, seed=42):
    """GLCT-CNN: citra waktu-frekuensi 128x128 -> prediksi per window -> agregasi."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = _train_torch(CognitiveLoadCNN(2), Htr[:, None], ytr,
                     epochs=15, bs=C1.BATCH_SIZE, lr=C1.LEARNING_RATE, wd=C1.WEIGHT_DECAY)
    return _aggregate(_proba(m, Hnew[:, None]), act_of_window, acts)


def predict_lstm(Xtr, ytr, Xnew, act_of_window, acts, seed=42):
    """AttnBiLSTM: deret waktu 600 titik -> prediksi per window -> agregasi."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = _train_torch(AttnBiLSTM(), Xtr, ytr, epochs=15)
    return _aggregate(_proba(m, Xnew), act_of_window, acts)


def heatmaps_for(windows, tag):
    """Heatmap GLCT untuk window subjek baru (SIGNAL_NORM='none' -> tak butuh stats)."""
    X = np.asarray(windows, dtype=np.float32)
    return build_heatmaps(X, groups=np.zeros(len(X), int),
                          subj_stats=np.array([[0.0, 1.0]], dtype=np.float32),
                          apply_hpf=True, tag=tag, verbose=False)
