"""Dasbor edukatif: napak tilas SELURUH pipeline penelitian, dari sinyal GSR
mentah sampai perbandingan akhir keempat model.

Berbeda dari app.py (demo prediksi interaktif untuk CSV baru), berkas ini
MENJELASKAN tiap tahap memakai data nyata dari penelitian: bentuk data mentah,
efek tiap langkah praproses, cara pelabelan NASA-TLX, isi tabel fitur, bentuk
tiap representasi (tabular/citra/deret/graf), dan hasil evaluasi final yang
TERKUNCI (pipeline_lengkap/, dihitung ulang dari fungsi pipeline asli lewat
prepare_eval.py -> eval_cache.npz).

Jalankan:  streamlit run app_streamlit/showcase.py
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

from experiments.shared import eda_preprocessing as E
from experiments.shared.config_shared import (
    FS, GSR_COLUMN, TIME_COLUMN, ACTIVITY_CODE_COLUMN, ACTIVITY_CODE_MAP,
    MAX_INTERP_SEC, BASELINE_NORM, DATA_DIR, WWL_CSV, DETAIL_CSV)
from experiments.shared.data import _resample_uniform, list_files
from experiments.shared.labels_nasatlx import norm_name

# Identitas pada repo sudah dianonimkan (`r_1`, `r_2`, dan seterusnya). Dua
# rekaman milik responden yang sama sudah memakai ID anonim yang sama.
FILE_ALIAS = {}

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(ROOT, "dataset_fitur")
KELAS = ["Rendah", "Tinggi"]
WARNA = {"Rendah": "#1565c0", "Tinggi": "#ef6c00"}
ACT_COLOR = {"watching": "#1565c0", "reading": "#2e7d32",
             "discussion": "#8e24aa", "quiz": "#c62828"}
# Warna per KODE aktivitas (bukan teks LMS_Activity, yang penulisannya tak selalu
# konsisten) -> sumber warna jadi sama persis dgn ACTIVITY_CODE_MAP yang dipakai
# pipeline, sehingga tiap aktivitas pasti kebagian warna.
CODE_COLOR = {0: "#9e9e9e", **{c: ACT_COLOR[a] for c, a in ACTIVITY_CODE_MAP.items()}}
MODEL_KEYS = {"Gradient Boosting": "gradboost", "GCN": "gcn",
              "GLCT-CNN": "glct_cnn", "AttnBiLSTM": "attnbilstm"}

st.set_page_config(page_title="Muhammad Satria Dharma - Penelitian", layout="wide")

# ===================================================================== data
@st.cache_data
def list_raw_files():
    files = list_files()
    return [f for f in files if not os.path.basename(f).startswith("_")]


def subject_from_file(path):
    m = re.match(r"(.+?)_\d{8}_", os.path.basename(path))
    raw = m.group(1) if m else os.path.basename(path).replace(".csv", "")
    return FILE_ALIAS.get(raw, raw)


@st.cache_data
def build_responden_registry():
    """Satu responden -> daftar berkas CSV miliknya (biasanya 1, kadang 2)."""
    reg = {}
    for f in list_raw_files():
        reg.setdefault(subject_from_file(f), []).append(f)
    return dict(sorted(reg.items(), key=lambda kv: kv[0].lower()))


@st.cache_data
def read_raw(path):
    return pd.read_csv(path)


@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(FIGDIR, "features_extended.csv"))


@st.cache_data
def load_wwl():
    return pd.read_csv(WWL_CSV)


@st.cache_data
def load_detail_weights():
    return pd.read_csv(DETAIL_CSV)


@st.cache_resource
def load_nodes_npz():
    d = np.load(os.path.join(HERE, "windows_train.npz"), allow_pickle=True)
    return dict(Xn=d["Xn"], yn=d["yn"], node_names=[str(x) for x in d["node_names"]],
                Xraw=d["Xraw"], H=d["H"], y=d["y"], meta=[str(x) for x in d["meta"]],
                feat_names=[str(x) for x in d["feature_names"]])


@st.cache_resource
def load_eval_cache():
    path = os.path.join(HERE, "eval_cache.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    out = {}
    for label, key in MODEL_KEYS.items():
        out[label] = dict(AL=d[f"{key}_AL"], AP=d[f"{key}_AP"],
                          PR=d[f"{key}_PR"], fa=d[f"{key}_fa"])
    return out


def compute_metrics(AL, AP, PR):
    return dict(
        akurasi=accuracy_score(AL, AP),
        presisi=precision_score(AL, AP, zero_division=0),
        recall=recall_score(AL, AP, zero_division=0),
        f1=f1_score(AL, AP, zero_division=0),
        auc=roc_auc_score(AL, PR))


def resolve_file(files, key):
    """Kembalikan path berkas responden; bila >1 berkas, tampilkan selectbox."""
    if len(files) == 1:
        return files[0]
    idx = st.selectbox("Berkas", range(len(files)),
                       format_func=lambda i: os.path.basename(files[i]), key=key)
    return files[idx]


D = load_nodes_npz()
P33, P67 = 48.9, 63.1

# ===================================================================== sidebar
st.sidebar.title("Visualisasi Penelitian")
st.sidebar.caption("Napak tilas pipeline SATU responden, dari sinyal mentah "
                    "sampai representasi akhir (saran dosen pembimbing).")
REG = build_responden_registry()
RESP = st.sidebar.selectbox("Responden", list(REG.keys()))
st.sidebar.divider()
page = st.sidebar.radio("Tahap", [
    "Data Mentah",
    "Praproses Sinyal",
    "Pelabelan (NASA-TLX)",
    "Ekstraksi Fitur",
    "Empat Representasi",
    "Hasil Evaluasi Model",
    "Perbandingan",
])

def hr(txt):
    st.markdown(f"### {txt}")


# ===================================================================== DATA MENTAH
if page == "Data Mentah":
    st.title("Data Mentah")
    st.markdown(
        "Setiap sesi responden direkam oleh sensor GSR (Grove) melalui aplikasi LMS, "
        "disimpan sebagai satu CSV per subjek. Kolomnya:")
    st.table(pd.DataFrame({
        "Kolom": ["TimeStamp", "gsrRaw", "LMS_Activity", "Activity_Code"],
        "Arti": ["Waktu perekaman",
                 "Nilai ADC mentah sensor GSR (bukan µS)",
                 "Label aktivitas LMS saat itu (teks)",
                 "Kode numerik aktivitas: 0 kalibrasi/istirahat/nasa-tlx, "
                 "1 watching, 2 reading, 3 discussion, 4 quiz"],
    }))

    path = resolve_file(REG[RESP], key="dm_file")
    df = read_raw(path)
    st.caption(f"Responden: **{RESP}** — berkas `{os.path.basename(path)}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Data Row", f"{len(df):,}")
    c2.metric("Durasi rekaman", f"{(pd.to_datetime(df[TIME_COLUMN]).iloc[-1] - pd.to_datetime(df[TIME_COLUMN]).iloc[0]).total_seconds()/60:.1f} menit")
    c3.metric("Aktivitas terekam", df[ACTIVITY_CODE_COLUMN].nunique())

    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("**Sinyal `gsrRaw` mentah, diwarnai per aktivitas:**")
    ts = pd.to_datetime(df[TIME_COLUMN])
    t = (ts - ts.iloc[0]).dt.total_seconds().values
    fig, ax = plt.subplots(figsize=(11, 3.2))
    codes = df[ACTIVITY_CODE_COLUMN].values
    gsr = df[GSR_COLUMN].values
    prev = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or codes[i] != codes[i - 1]:
            ax.plot(t[prev:i], gsr[prev:i], lw=.7, color=CODE_COLOR.get(codes[prev], "#9e9e9e"))
            prev = i
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=a) for a, c in ACT_COLOR.items()]
    handles.append(plt.Line2D([0], [0], color="#9e9e9e", lw=3, label="kalibrasi/istirahat"))
    ax.legend(handles=handles, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    ax.set_xlabel("waktu (detik)"); ax.set_ylabel("gsrRaw (ADC)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ===================================================================== PRAPROSES
elif page == "Praproses Sinyal":
    st.title("Praproses Sinyal (langkah demi langkah)")

    path = resolve_file(REG[RESP], key="pp_file")
    df = read_raw(path)
    st.caption(f"Responden: **{RESP}** — berkas `{os.path.basename(path)}`")
    act_opts = [a for c, a in ACTIVITY_CODE_MAP.items() if (df[ACTIVITY_CODE_COLUMN] == c).any()]
    if not act_opts:
        st.error("Berkas ini tidak punya aktivitas watching/reading/discussion/quiz yang lengkap.")
        st.stop()
    act = st.selectbox("Aktivitas", act_opts)
    code = [c for c, a in ACTIVITY_CODE_MAP.items() if a == act][0]

    ts = pd.to_datetime(df[TIME_COLUMN])
    t = (ts - ts.iloc[0]).dt.total_seconds().values
    m = df[ACTIVITY_CODE_COLUMN].values == code
    if m.sum() < 20:
        st.warning("Segmen terlalu pendek untuk didemokan.")
        st.stop()
    gsr = df[GSR_COLUMN].values.astype(float)

    x0 = _resample_uniform(t[m] - t[m][0], gsr[m], FS)
    x1, valid = E.clean_dropouts(x0, FS, MAX_INTERP_SEC)
    x2 = E.clean_nk(x1, FS)
    all_clean = []
    for c2, a2 in ACTIVITY_CODE_MAP.items():
        mm = df[ACTIVITY_CODE_COLUMN].values == c2
        if mm.sum() < 20:
            continue
        xr = _resample_uniform(t[mm] - t[mm][0], gsr[mm], FS)
        xc, _ = E.clean_dropouts(xr, FS, MAX_INTERP_SEC)
        all_clean.append(E.clean_nk(xc, FS))
    stats = E.subject_norm_stats(np.concatenate(all_clean), BASELINE_NORM)
    x3 = E.baseline_normalize(x2, BASELINE_NORM, stats)
    tonic, phasic = E.split_tonic_phasic(x3, FS)

    # Skala ADC (x0-x2) dikunci SAMA lintas-plot supaya efek tiap langkah kelihatan
    # nyata (bukan tersamar auto-scale per-plot); langkah SEBELUMNYA ditumpangkan
    # tipis abu-abu di belakang langkah SEKARANG supaya bedanya langsung terlihat.
    adc_lo = min(x0.min(), x1.min(), x2.min())
    adc_hi = max(x0.max(), x1.max(), x2.max())
    pad = (adc_hi - adc_lo) * 0.08

    steps = [
        ("Mentah (resampled 10 Hz)", x0, "#9e9e9e", "gsrRaw (ADC)", None),
        ("Dropout dibersihkan", x1, "#5c6bc0", "gsrRaw (ADC)", x0),
        ("Butterworth low-pass 3 Hz orde 4 (NeuroKit2)", x2, "#00897b", "gsrRaw (ADC)", x1),
        ("Normalisasi baseline z-score per-subjek", x3, "#1565c0", "z-score", None),
    ]
    for title, sig, color, ylabel, prev in steps:
        st.markdown(f"**{title}**")
        c1, c2 = st.columns([2, 1]) if prev is not None else (st, None)
        fig, ax = plt.subplots(figsize=(11 if prev is None else 7.3, 2))
        if prev is not None:
            ax.plot(np.arange(len(prev)) / FS, prev, lw=.8, color="#cfd8dc", label="sebelum")
            ax.legend(fontsize=7, loc="upper right")
        ax.plot(np.arange(len(sig)) / FS, sig, lw=1.1, color=color)
        if ylabel == "gsrRaw (ADC)":
            ax.set_ylim(adc_lo - pad, adc_hi + pad)
        ax.set_xlabel("waktu (detik)"); ax.set_ylabel(ylabel)
        fig.tight_layout(); c1.pyplot(fig); plt.close(fig)

        if prev is not None:
            # Perbedaan tiap langkah (interpolasi 1 gap, atau smoothing derau halus)
            # kecil dibanding durasi 1 aktivitas penuh, jadi tersamar di plot full-width
            # di atas -> perbesar 16 detik di sekitar titik yang berubah PALING banyak.
            diff = np.abs(sig - prev)
            center = int(np.argmax(diff))
            lo = max(0, center - 8 * FS); hi = min(len(sig), center + 8 * FS)
            tt = np.arange(lo, hi) / FS
            figz, axz = plt.subplots(figsize=(3.6, 2))
            axz.plot(tt, prev[lo:hi], lw=1.2, color="#cfd8dc", label="sebelum")
            axz.plot(tt, sig[lo:hi], lw=1.4, color=color, label="sesudah")
            axz.set_title("perbesar area paling berubah", fontsize=8)
            axz.legend(fontsize=7); axz.set_xlabel("waktu (detik)")
            figz.tight_layout(); c2.pyplot(figz); plt.close(figz)

        if title.startswith("Normalisasi"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rerata sebelum", f"{x2.mean():.1f}")
            m2.metric("Rerata sesudah", f"{x3.mean():.2f}")
            m3.metric("Std sebelum", f"{x2.std():.1f}")
            m4.metric("Std sesudah", f"{x3.std():.2f}")

    st.markdown("**Dekomposisi tonik (SCL) / fasik (SCR), cutoff 0,05 Hz**")
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(np.arange(len(x3)) / FS, tonic, lw=1.3, color="#ef6c00", label="tonik (SCL)")
    ax.plot(np.arange(len(x3)) / FS, phasic, lw=.7, color="#6a1b9a", label="fasik (SCR)", alpha=.8)
    ax.legend(fontsize=8); ax.set_xlabel("waktu (detik)"); ax.set_ylabel("z-score")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ===================================================================== PELABELAN
elif page == "Pelabelan (NASA-TLX)":
    st.title("Pelabelan: NASA-TLX → WWL → Extreme Groups Approach")

    hr("Skor WWL per aktivitas")
    dw = load_detail_weights()
    mine = dw[dw["Respondent"].map(norm_name) == norm_name(RESP)].reset_index(drop=True)
    if mine.empty:
        st.info(f"**{RESP}** tidak memiliki data kuesioner NASA-TLX.")
    else:
        ridx = st.selectbox("Aktivitas", mine.index, format_func=lambda i: mine.loc[i, "Activity"])
        row = mine.loc[ridx]
        dims = ["MD", "PD", "TD", "OP", "FR", "EF"]
        dim_names = {"MD": "Mental Demand", "PD": "Physical Demand", "TD": "Temporal Demand",
                     "OP": "Performance", "FR": "Frustration", "EF": "Effort"}
        tbl = pd.DataFrame({
            "Dimensi": [dim_names[d] for d in dims],
            "Rating": [row[f"Rating_{d}"] for d in dims],
            "Weight": [row[f"Weight_{d}"] for d in dims],
            "Rating × Weight": [row[f"WC_{d}"] for d in dims],
        })
        st.table(tbl)
        st.markdown(f"Total = {row['Total_Weighted']:.2f} → **WWL = {row['Total_Weighted']:.2f} / 15 "
                    f"= {row['WWL']:.2f}**")

    hr("Sebaran WWL & ambang Extreme Groups Approach")
    wwl = load_wwl()
    long = wwl.melt(id_vars="Respondent", value_vars=list(ACTIVITY_CODE_MAP.values()),
                     var_name="activity", value_name="wwl").dropna(subset=["wwl"])
    mine_long = long[long["Respondent"].map(norm_name) == norm_name(RESP)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.hist(long["wwl"], bins=20, color="#e0e0e0", edgecolor="white")
    ax.axvline(P33, color="#1565c0", ls="--", lw=1.6, label=f"P33 = {P33}")
    ax.axvline(P67, color="#c62828", ls="--", lw=1.6, label=f"P67 = {P67}")
    ymax = ax.get_ylim()[1]
    for _, r in mine_long.iterrows():
        ax.axvline(r["wwl"], color=ACT_COLOR.get(r["activity"], "#000"), lw=2)
        ax.annotate(r["activity"], (r["wwl"], ymax * 0.95), rotation=90,
                    fontsize=7, ha="right", va="top")
    ax.set_xlabel("WWL"); ax.set_ylabel("jumlah aktivitas"); ax.legend(fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if not mine_long.empty:
        st.caption(f"Garis berwarna: posisi WWL tiap aktivitas **{RESP}** di antara seluruh responden.")

    st.markdown(
        f"**Aturan EGA**: `WWL ≤ {P33}` → **Rendah** (0) · `WWL ≥ {P67}` → **Tinggi** (1).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Node awal (semua aktivitas)", "83")
    c2.metric("Dibuang (tertil tengah)", "29")
    c3.metric("Node Rendah", "27")
    c4.metric("Node Tinggi", "27")

# ===================================================================== FITUR
elif page == "Ekstraksi Fitur":
    st.title("Ekstraksi Fitur")
    st.markdown("Dari tiap window 60 detik (sinyal ternormalisasi + dekomposisi tonik/fasik) "
                "diekstrak fitur EDA: tonik, fasik/SCR, distribusi sinyal, dinamika, "
                "frekuensi, dan Hjorth.")

    df = load_features()
    view = df[df["subject"].map(norm_name) == norm_name(RESP)]
    if view.empty:
        st.info(f"**{RESP}** tidak masuk himpunan fitur (tak lolos pelabelan EXTREME 60 detik).")
    else:
        act_opts = ["(semua)"] + sorted(view["activity"].unique())
        act = st.selectbox("Aktivitas", act_opts)
        if act != "(semua)":
            view = view[view["activity"] == act]
        st.caption(f"{len(view)} baris (window) × {view.shape[1]} kolom milik **{RESP}**.")
        st.dataframe(view, use_container_width=True)

# ===================================================================== REPRESENTASI
elif page == "Empat Representasi":
    st.title("Empat Representasi Data")
    st.markdown(f"Sinyal **{RESP}** yang SAMA diubah menjadi empat bentuk berbeda, satu per model.")

    st.table(pd.DataFrame({
        "Paradigma": ["Tabular", "Graf", "Citra", "Deret waktu"],
        "Model": ["Gradient Boosting", "GCN (transduktif)", "GLCT-CNN", "AttnBiLSTM"],
        "Bentuk data": ["vektor 37 fitur / node aktivitas", "graf k-NN (k=8) atas vektor fitur",
                         "citra 128×128 (transform GLCT)", "deret 600 titik (60 detik @10 Hz)"],
        "Satuan penilaian": ["node (aktivitas)", "node (aktivitas)", "window", "window"],
    }))

    key = norm_name(RESP)
    node_idx = [i for i, n in enumerate(D["node_names"]) if norm_name(n.split("|")[0]) == key]
    win_idx = [i for i, m in enumerate(D["meta"]) if norm_name(m.split("|")[0]) == key]
    not_in_training = (f"**{RESP}** tidak masuk 20 subjek data latih model (tak lolos pelabelan "
                       "EXTREME), sehingga representasi ini tidak tersedia untuknya.")

    t1, t2, t3, t4 = st.tabs(["Tabular", "Graf (GCN)", "Citra (GLCT-CNN)", "Deret (AttnBiLSTM)"])

    with t1:
        if not node_idx:
            st.info(not_in_training)
        else:
            i = st.selectbox("Aktivitas", node_idx,
                             format_func=lambda i: f"{D['node_names'][i].split('|')[1]} — {KELAS[D['yn'][i]]}")
            st.dataframe(pd.DataFrame({"fitur": D["feat_names"], "nilai": D["Xn"][i].round(3)}),
                         use_container_width=True, height=300)

    with t2:
        from sklearn.preprocessing import StandardScaler

        def knn_adj(F, k=8):
            Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            S = Fn @ Fn.T; np.fill_diagonal(S, -1)
            A = np.zeros_like(S)
            for r in range(len(S)):
                A[r, np.argsort(S[r])[::-1][:k]] = 1
            return np.maximum(A, A.T)

        Xs = StandardScaler().fit_transform(D["Xn"])
        A = knn_adj(Xs, 8)
        G = nx.from_numpy_array(A)
        pos = nx.spring_layout(G, seed=42, k=0.5)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [WARNA[KELAS[c]] for c in D["yn"]]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=.25, width=.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=90,
                                edgecolors="black", linewidths=.4)
        if node_idx:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_idx,
                                   node_color=[WARNA[KELAS[D["yn"][i]]] for i in node_idx],
                                   node_size=420, node_shape="*",
                                   edgecolors="black", linewidths=1.2)
        ax.set_title(f"{len(D['Xn'])} node, {G.number_of_edges()} sisi — biru=Rendah, oranye=Tinggi"
                    + (f" — bintang = {RESP}" if node_idx else ""))
        ax.axis("off")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        if not node_idx:
            st.info(not_in_training)
        homofili = np.mean([D["yn"][u] == D["yn"][v] for u, v in G.edges()])
        st.caption(f"Homofili graf (sisi yang menghubungkan label SAMA): {homofili:.1%}.")

    with t3:
        if not win_idx:
            st.info(not_in_training)
        else:
            j = st.selectbox("Window", win_idx,
                             format_func=lambda j: f"{D['meta'][j].split('|')[1]} — {KELAS[D['y'][j]]}",
                             key="repr_img")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(D["H"][j], cmap="inferno", aspect="auto")
            ax.set_xlabel("waktu"); ax.set_ylabel("frekuensi (chirp rate)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with t4:
        if not win_idx:
            st.info(not_in_training)
        else:
            k = st.selectbox("Window", win_idx,
                             format_func=lambda k: f"{D['meta'][k].split('|')[1]} — {KELAS[D['y'][k]]}",
                             key="repr_seq")
            fig, ax = plt.subplots(figsize=(9, 2.6))
            ax.plot(np.arange(D["Xraw"].shape[1]) / FS, D["Xraw"][k], lw=.9,
                    color=WARNA[KELAS[D["y"][k]]])
            ax.set_xlabel("waktu (detik)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ===================================================================== HASIL
elif page == "Hasil Evaluasi Model":
    st.title("Hasil Evaluasi Model")
    st.caption("Evaluasi bersifat lintas-subjek (GroupKFold-5 atas seluruh 21 responden), "
               "sehingga tidak direduksi ke satu responden seperti tahap sebelumnya — "
               "mengukur generalisasi model butuh banyak subjek sekaligus.")

    cache = load_eval_cache()
    if cache is None:
        st.error("`eval_cache.npz` belum ada. Jalankan `python -m app_streamlit.prepare_eval` "
                 "sekali untuk menghitungnya dari pipeline final terkunci.")
        st.stop()

    model = st.selectbox("Pilih model", list(cache.keys()))
    c = cache[model]
    met = compute_metrics(c["AL"], c["AP"], c["PR"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{met['akurasi']:.3f}")
    c2.metric("Precision", f"{met['presisi']:.3f}")
    c3.metric("Recall", f"{met['recall']:.3f}")
    c4.metric("F1-Score", f"{met['f1']:.3f}")

    hr("Hasil per fold")
    fold_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(c["fa"]))],
                            "Akurasi": [f"{v:.3f}" for v in c["fa"]]})
    st.table(fold_df.set_index("Fold"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(c["AL"], c["AP"])
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for cc in range(2):
                ax.text(cc, r, cm[r, cc], ha="center", va="center",
                        color="white" if cm[r, cc] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Rendah", "Tinggi"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Rendah", "Tinggi"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown("**Kurva ROC**")
        fpr, tpr, _ = roc_curve(c["AL"], c["PR"])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(fpr, tpr, color="#1565c0", lw=2, label=f"AUC = {met['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="#bdbdbd", lw=1, ls="--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ===================================================================== PERBANDINGAN
elif page == "Perbandingan":
    st.title("Perbandingan Keseluruhan")

    cache = load_eval_cache()
    if cache is None:
        st.error("`eval_cache.npz` belum ada. Jalankan `python -m app_streamlit.prepare_eval` "
                 "sekali untuk menghitungnya dari pipeline final terkunci.")
        st.stop()

    rows = []
    for name, c in cache.items():
        met = compute_metrics(c["AL"], c["AP"], c["PR"])
        rows.append({"Model": name, "Akurasi": met["akurasi"], "F1-Score": met["f1"],
                     "Precision": met["presisi"], "Recall": met["recall"], "ROC-AUC": met["auc"]})
    tdf = pd.DataFrame(rows)
    disp = tdf.copy()
    for col in ["Akurasi", "F1-Score", "Precision", "Recall", "ROC-AUC"]:
        disp[col] = disp[col].map(lambda v: f"{v:.3f}")
    st.table(disp.set_index("Model"))

    hr("Perbandingan metrik antar-model")
    metric_cols = ["Akurasi", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(tdf)); n = len(metric_cols); w = 0.8 / n
    palette = ["#1565c0", "#00897b", "#8e24aa", "#ef6c00", "#546e7a"]
    for j, col in enumerate(metric_cols):
        ax.bar(x + (j - n/2 + 0.5) * w, tdf[col], w, label=col, color=palette[j])
    ax.set_xticks(x); ax.set_xticklabels(tdf["Model"])
    ax.axhline(0.5, color="#bdbdbd", lw=.8, ls="--")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_ylim(0, 1)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    hr("Confusion matrix setiap model")
    cols = st.columns(len(cache))
    for col, (name, c) in zip(cols, cache.items()):
        cm = confusion_matrix(c["AL"], c["AP"])
        fig, ax = plt.subplots(figsize=(3.2, 3))
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for cc in range(2):
                ax.text(cc, r, cm[r, cc], ha="center", va="center",
                        color="white" if cm[r, cc] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Rendah", "Tinggi"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Rendah", "Tinggi"], fontsize=8)
        ax.set_xlabel("Prediksi", fontsize=8); ax.set_ylabel("Sebenarnya", fontsize=8)
        ax.set_title(name, fontsize=10)
        fig.tight_layout(); col.pyplot(fig); plt.close(fig)
