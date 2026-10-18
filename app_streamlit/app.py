"""Aplikasi demo: klasifikasi beban kognitif lintas-subjek dari sinyal GSR.

EMPAT model, satu satuan penilaian yang sama (per aktivitas):
  SVM         (induktif)    fitur statistik  -> node aktivitas
  GLCT-CNN    (induktif)    citra waktu-frekuensi -> per window -> AGREGASI
  AttnBiLSTM  (induktif)    deret waktu 1-D  -> per window -> AGREGASI
  GCN         (TRANSDUKTIF) graf k-NN        -> node aktivitas

Pengaman kejujuran:
  1. Bila subjek yang diuji ADA di data latih, seluruh datanya DIKELUARKAN lebih
     dulu, lalu keempat model DILATIH ULANG. Tanpa ini akurasinya palsu.
  2. Aktivitas yang WWL-nya di TERTIL TENGAH tak punya kebenaran dasar, jadi tidak
     diikutkan dalam perhitungan akurasi (tetap ditampilkan prediksinya).
  3. CNN & AttnBiLSTM memakai epoch TETAP (tanpa early stopping), sehingga lipatan
     uji tak pernah diintip.

Jalankan:  streamlit run app_streamlit/app.py
"""
import os
import sys
import time

# `streamlit run` menaruh folder SKRIP di sys.path, bukan akar proyek, sehingga
# paket `app_streamlit` dan `experiments` tak ditemukan. Tambahkan akar proyek dulu.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from app_streamlit import core
from experiments.shared import eda_preprocessing as E
from experiments.shared.config_shared import FS

HERE = os.path.dirname(__file__)
KELAS = core.KELAS
WARNA = {"Rendah": "#2e7d32", "Tinggi": "#c62828"}

st.set_page_config(page_title="Klasifikasi Beban Kognitif GSR", layout="wide")


@st.cache_resource
def load_data():
    n = np.load(os.path.join(HERE, "nodes_train.npz"), allow_pickle=True)
    w = np.load(os.path.join(HERE, "windows_train.npz"), allow_pickle=True)
    wwl = {}
    for key, value in zip(n["wwl_keys"], n["wwl_vals"].tolist()):
        subject, activity = str(key).split("|", 1)
        wwl[f"{core.norm(subject)}|{activity}"] = value
    return dict(
        Xn=w["Xn"], yn=w["yn"], node_names=[str(x) for x in w["node_names"]],
        Xraw=w["Xraw"], H=w["H"], y=w["y"], meta=[str(x) for x in w["meta"]],
        subjects=[str(s) for s in w["subjects"]],
        feat_names=[str(f) for f in w["feature_names"]],
        p33=float(n["p33"]), p67=float(n["p67"]), wwl=wwl)


D = load_data()

st.title("Klasifikasi Beban Kognitif Lintas-Subjek Berbasis GSR")
st.caption(
    f"Data latih: {len(D['Xn'])} aktivitas ({len(D['Xraw'])} window) dari "
    f"{len(D['subjects'])} responden. Base rate {max(np.bincount(D['yn']))/len(D['yn']):.3f}. "
    f"Ambang tertil: Rendah bila WWL ≤ {D['p33']:.1f}, Tinggi bila WWL ≥ {D['p67']:.1f}.")

with st.sidebar:
    st.header("Masukan")
    up = st.file_uploader("Berkas CSV rekaman GSR", type="csv")
    st.markdown("**Kolom wajib:** `TimeStamp`, `gsrRaw`, `Activity_Code` "
                "(1 watching, 2 reading, 3 discussion, 4 quiz).")
    st.divider()
    mode = st.radio(
        "Mode",
        ["evaluasi", "penerapan"],
        format_func=lambda v: {
            "evaluasi": "Evaluasi (peneliti)",
            "penerapan": "Penerapan (pengguna baru)"}[v],
        help="Evaluasi: skor NASA-TLX responden dipakai sebagai KUNCI JAWABAN untuk "
             "mengoreksi model, sehingga akurasi bisa dihitung. Menjawab 'apakah modelnya "
             "bagus?'.\n\nPenerapan: NASA-TLX DISEMBUNYIKAN sepenuhnya, seolah pengguna "
             "belum pernah mengisi kuesioner. Menjawab 'apa yang sebenarnya dilihat "
             "pengguna di lapangan?'.")
    if mode == "penerapan":
        st.caption("NASA-TLX disembunyikan. Tak ada akurasi, tak ada label sebenarnya, "
                   "tak ada peringatan tertil tengah. Persis seperti kondisi nyata.")
    st.divider()
    base = st.radio("Acuan normalisasi per-subjek", ["task", "rest"],
                    format_func=lambda v: {"task": "Seluruh sesi tugas",
                                           "rest": "Segmen istirahat"}[v])
    pakai_dl = st.checkbox("Sertakan GLCT-CNN & AttnBiLSTM", value=True,
                           help="Keduanya dilatih ulang dari nol (sekitar 1-2 menit). "
                                "Matikan bila ingin cepat (hanya SVM & GCN).")
    st.divider()
    st.caption("GCN dijalankan sebagai ensemble 10 seed. CNN & AttnBiLSTM memprediksi "
               "per window, lalu digabung per aktivitas lewat rata-rata probabilitas.")

if up is None:
    st.info("Unggah CSV pada panel kiri. Untuk mencoba subjek yang benar-benar belum "
            "pernah dilihat model, gunakan **r_8** (tersingkir dari data latih karena "
            "seluruh aktivitasnya jatuh di tertil tengah).")
    st.stop()

df = pd.read_csv(up)
if (probs := core.validate(df)):
    for p in probs:
        st.error(p)
    st.stop()

# ---------------------------------------------------------------- anti-bocor
subj = core.subject_from_filename(up.name)
key = core.norm(subj)
keep_n = [i for i, n in enumerate(D["node_names"]) if core.norm(n.split("|")[0]) != key]
keep_w = [i for i, m in enumerate(D["meta"]) if core.norm(m.split("|")[0]) != key]
Xn, yn = D["Xn"][keep_n], D["yn"][keep_n]
node_names = [D["node_names"][i] for i in keep_n]
Xraw, H, yw = D["Xraw"][keep_w], D["H"][keep_w], D["y"][keep_w]
n_buang = len(D["node_names"]) - len(keep_n)

if n_buang:
    st.warning(f"**{subj}** ada di data latih. {n_buang} aktivitas miliknya "
               f"**dikeluarkan**, tersisa {len(Xn)} aktivitas ({len(Xraw)} window) dari "
               f"{len(D['subjects'])-1} responden lain. Keempat model dilatih ulang dari "
               f"nol tanpa dia, sehingga akurasi yang terukur tetap jujur.")
else:
    st.success(f"**{subj}** tidak ada di data latih. Prediksi murni lintas-subjek "
               f"terhadap {len(Xn)} aktivitas dari {len(D['subjects'])} responden.")

with st.spinner("Praproses sinyal..."):
    segs, info = core.preprocess(df, baseline_ref=base)
valid = {a: s for a, s in segs.items() if s.get("node") is not None}
if not valid:
    st.error("Tidak ada aktivitas dengan window 60 detik yang bersih. Rekaman gagal.")
    st.stop()

acts = list(valid)
Xnew = np.array([valid[a]["node"] for a in acts])
win_new = np.array([w for a in acts for w in valid[a]["windows"]], dtype=np.float32)
act_of_win = [a for a in acts for _ in valid[a]["windows"]]

# Cocokkan ID berkas ke tabel NASA-TLX. Bedakan tegas TIGA keadaan:
#   wwl_key None  -> kuesioner TAK ADA        (akurasi tak bisa dihitung: tak ada kunci)
#   truth  None   -> WWL ada, TERTIL TENGAH   (akurasi tak bisa dihitung: kunci kosong)
#   truth  0/1    -> WWL ada, label jelas     (akurasi bisa dihitung)
wwl_key = core.resolve_wwl_subject(up.name, D["wwl"].keys())
ada_kuesioner = (mode == "evaluasi") and wwl_key is not None    # mode penerapan: buta
wwl = [D["wwl"].get(f"{wwl_key}|{a}") if ada_kuesioner else None for a in acts]
truth = [core.true_label(w, D["p33"], D["p67"]) for w in wwl]

# ---------------------------------------------------------------- prediksi
hasil, waktu = {}, {}
t = time.time(); l, c = core.predict_svm(Xn, yn, Xnew)
hasil["SVM"] = (l, [f"p={x:.2f}" for x in c]); waktu["SVM"] = time.time() - t

with st.spinner(f"GCN: bangun ulang graf {len(Xn)}+{len(acts)} node, latih ulang 10 seed..."):
    t = time.time(); l, v = core.predict_gcn(Xn, yn, Xnew)
    hasil["GCN"] = (l, [f"{int(v[i] if l[i]==1 else 10-v[i])}/10 seed" for i in range(len(l))])
    waktu["GCN"] = time.time() - t

if pakai_dl:
    with st.spinner("GLCT-CNN: hitung heatmap + latih ulang 15 epoch..."):
        t = time.time()
        Hnew = core.heatmaps_for(win_new, tag=f"app_{key}")
        l, c = core.predict_cnn(H, yw, Hnew, act_of_win, acts)
        hasil["GLCT-CNN"] = (l, [f"p={x:.2f}" for x in c]); waktu["GLCT-CNN"] = time.time() - t
    with st.spinner("AttnBiLSTM: latih ulang 15 epoch..."):
        t = time.time()
        l, c = core.predict_lstm(Xraw, yw, win_new, act_of_win, acts)
        hasil["AttnBiLSTM"] = (l, [f"p={x:.2f}" for x in c]); waktu["AttnBiLSTM"] = time.time() - t

MODEL = list(hasil)
tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Akurasi & Prediksi", "2. Sinyal", "3. Fitur", "4. Graf transduktif"])

with tab1:
    nilai = [i for i, t_ in enumerate(truth) if t_ is not None]
    rows = []
    for i, a in enumerate(acts):
        r = {"aktivitas": a, "window": len(valid[a]["windows"])}
        if ada_kuesioner:          # mode evaluasi: tampilkan kunci jawaban
            r["WWL asli"] = f"{wwl[i]:.1f}" if wwl[i] is not None else "-"
            r["label sebenarnya"] = (KELAS[truth[i]] if truth[i] is not None
                                     else "tertil tengah")
        for m in MODEL:
            lab, ket = hasil[m]
            r[m] = f"{KELAS[lab[i]]} ({ket[i]})"
            if ada_kuesioner:
                r[f"{m} ✓"] = ("BENAR" if lab[i] == truth[i] else "SALAH") \
                    if truth[i] is not None else "-"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if mode == "penerapan":
        st.error(
            "**Inilah yang sebenarnya dilihat pengguna di lapangan.** Tidak ada kolom WWL, "
            "tidak ada label sebenarnya, tidak ada akurasi, dan tidak ada peringatan apa pun. "
            "Keempat model menjawab dengan yakin untuk SETIAP aktivitas, termasuk aktivitas "
            "yang sebenarnya berbeban sedang dan tidak dapat dijawab. Pengguna tidak punya "
            "cara apa pun untuk tahu jawaban mana yang layak dipercaya, sebab untuk "
            "mengetahuinya ia perlu skor NASA-TLX, padahal skor itulah yang seharusnya "
            "diprediksi model.")
    elif wwl_key is None:
        st.info(f"**Akurasi tak dapat dihitung: {subj} tidak mengisi NASA-TLX**, sehingga "
                f"tidak ada kunci jawaban untuk mengoreksi model.")
    elif not nilai:
        st.error(f"**Akurasi tak dapat dihitung.** Seluruh aktivitas {subj} jatuh di tertil "
                 f"tengah (WWL antara {D['p33']:.1f} dan {D['p67']:.1f}), sehingga tak punya "
                 f"label sebenarnya. Model tetap menjawab dengan yakin, dan justru itulah "
                 f"kelemahan pokoknya.")
    else:
        cols = st.columns(len(MODEL) + 1)
        cols[0].metric("Aktivitas dinilai", f"{len(nilai)} / {len(acts)}")
        for j, m in enumerate(MODEL):
            lab = hasil[m][0]
            acc = float(np.mean([lab[i] == truth[i] for i in nilai]))
            cols[j + 1].metric(m, f"{acc:.0%}", delta=f"{acc-0.519:+.0%} vs base")
        st.caption("Akurasi dari sedikit aktivitas, jadi sangat berderau. Jangan menyimpulkan "
                   "apa pun dari satu responden. Acuan menyeluruh (GroupKFold-5, 20 responden, "
                   "**bebas kebocoran**): SVM ~0,61 · GCN ~0,48-0,61 · GLCT-CNN ~0,61 · "
                   "AttnBiLSTM ~0,58 · base rate 0,519.")

    if len(acts) - len(nilai):
        st.info(f"{len(acts)-len(nilai)} aktivitas ada di tertil tengah, tidak diikutkan "
                f"dalam perhitungan akurasi.")

    st.divider()
    c1, c2 = st.columns(2)
    c1.markdown("**Induktif** (model dapat dikirim sendirian sebagai satu berkas)")
    c1.markdown(f"- SVM, {waktu['SVM']:.2f} detik" +
                (f"\n- GLCT-CNN, {waktu.get('GLCT-CNN',0):.0f} detik"
                 f"\n- AttnBiLSTM, {waktu.get('AttnBiLSTM',0):.0f} detik" if pakai_dl else ""))
    c2.markdown("**Transduktif** (WAJIB membawa seluruh data latih)")
    c2.markdown(f"- GCN, {waktu['GCN']:.2f} detik. Graf dibangun ulang dan model dilatih "
                f"ulang 10 kali setiap ada subjek baru. Tidak ada berkas model yang tetap.")
    st.caption("GLCT-CNN dan AttnBiLSTM memprediksi per window, lalu digabung per aktivitas "
               "lewat rata-rata probabilitas. Dengan begitu keempat model dinilai pada "
               "satuan yang sama, dan aktivitas berdurasi panjang tidak berbobot lebih besar.")

with tab2:
    for a in acts:
        s = valid[a]
        st.subheader(a)
        c1, c2, c3 = st.columns(3)
        c1.metric("Durasi", f"{s['durasi']:.0f} detik")
        c2.metric("Window dipakai", len(s["windows"]))
        c3.metric("Window ditolak", s["n_reject"])
        tonic, phasic = E.split_tonic_phasic(s["signal"], FS)
        tt = np.arange(len(s["signal"])) / FS
        fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
        ax[0].plot(tt, s["signal"], lw=.8, color="#1565c0")
        ax[0].set_ylabel("GSR (z-score)")
        ax[1].plot(tt, tonic, lw=1.2, color="#ef6c00", label="tonik")
        ax[1].plot(tt, phasic, lw=.8, color="#6a1b9a", label="fasik")
        ax[1].set_xlabel("waktu (detik)"); ax[1].legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab3:
    st.caption("Fitur paling menentukan menurut uji-F: tonic_range, tonic_std, sig_mad, "
               "sig_std, sig_iqr. Semuanya mengukur SEBARAN, bukan jumlah puncak SCR.")
    tab = pd.DataFrame({a: valid[a]["node"] for a in acts}, index=D["feat_names"]).round(3)
    tab["rerata data latih"] = Xn.mean(0).round(3)
    st.dataframe(tab, use_container_width=True, height=430)

with tab4:
    st.caption("Mekanisme transduksi: node baru (bintang) masuk ke dalam graf, lalu labelnya "
               "ditentukan oleh tetangga k-NN MILIK ORANG LAIN.")
    nb = core.neighbors(Xn, Xnew, k=8)
    V = np.vstack([Xn, Xnew])
    Z = PCA(2, random_state=0).fit_transform(V / (np.abs(V).max(0) + 1e-9))
    Zt, Zb = Z[:len(Xn)], Z[len(Xn):]
    gl = hasil["GCN"][0]
    fig, ax = plt.subplots(figsize=(9, 6))
    for cls, w_ in ((0, "#2e7d32"), (1, "#c62828")):
        m = yn == cls
        ax.scatter(Zt[m, 0], Zt[m, 1], c=w_, s=55, alpha=.55, edgecolor="none",
                   label=f"data latih: {KELAS[cls]}")
    for i, a in enumerate(acts):
        for j in nb[i]:
            Zj = Zt[j] if j < len(Xn) else Zb[j - len(Xn)]
            ax.plot([Zb[i, 0], Zj[0]], [Zb[i, 1], Zj[1]], color="#455a64", lw=.8,
                    alpha=.45, zorder=0)
        ax.scatter(*Zb[i], marker="*", s=460, c=WARNA[KELAS[gl[i]]], edgecolor="black",
                   linewidth=1.1, zorder=3)
        ax.annotate(a, Zb[i], xytext=(9, 7), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.scatter([], [], marker="*", s=200, c="white", edgecolor="black", label=f"aktivitas {subj}")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Graf k-NN (proyeksi PCA): node baru dan tetangganya")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    for i, a in enumerate(acts):
        tet = [node_names[j] for j in nb[i] if j < len(Xn)]
        hi = sum(int(yn[j]) for j in nb[i] if j < len(Xn))
        st.markdown(f"**{a}** → tetangga: `{', '.join(tet)}` ({hi} Tinggi, {len(tet)-hi} Rendah)")

st.divider()
st.warning(
    "**Batas penafsiran.** Prediksi bersifat RELATIF terhadap sebaran sampel penelitian ini "
    "(tertil bawah dan atas), bukan ukuran beban kognitif mutlak. Pada label rentang penuh "
    "akurasinya turun ke sekitar 0,48, di bawah tebakan acak. Keluaran berlaku untuk SATU "
    "AKTIVITAS UTUH, bukan pemantauan beban dari waktu ke waktu. Penelitian ini juga "
    "menemukan bahwa GSR merespons gairah fisik (gerak, bicara), bukan beban kognitif: pada "
    "dataset kontrol PhyAAt dengan manipulasi beban yang terverifikasi sangat kuat, GSR tetap "
    "tidak dapat mengklasifikasi (0,518 dari base rate 0,503).")
