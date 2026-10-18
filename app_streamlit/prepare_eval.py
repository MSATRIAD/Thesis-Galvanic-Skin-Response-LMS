"""Precompute evaluasi lengkap (per-fold, OOF prediksi+probabilitas, confusion)
dari KEEMPAT pipeline final TERKUNCI, disimpan ke eval_cache.npz supaya
showcase.py tak perlu melatih ulang model tiap kali dibuka.

Memanggil LANGSUNG fungsi asli tiap pipeline_final_*.py (prepare/evaluate),
bukan implementasi ulang -- angka dijamin identik dgn notebook evaluasi.

Jalankan sekali:  python -m app_streamlit.prepare_eval
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "eval_cache.npz")


def run_gradboost():
    from pipeline_lengkap.pipeline_final_gradboost import load_windows, build_nodes, evaluate
    import pandas as pd
    from experiments.shared.labels_nasatlx import norm_name
    Xraw, groups, meta, subs = load_windows()
    Xn, gn, names = build_nodes(Xraw, groups, meta)
    wtab = pd.read_csv("nasa_tlx_results/summary_wwl.csv")
    ACT_MAP = {1: "watching", 2: "reading", 3: "discussion", 4: "quiz"}
    W = {(norm_name(r.Respondent), a): r[a] for _, r in wtab.iterrows()
         for a in ACT_MAP.values() if pd.notna(r[a])}
    wwl = np.array([W.get((norm_name(n.split("|")[0]), n.split("|")[1]), np.nan) for n in names])
    ok = ~np.isnan(wwl); Xn, gn, wwl = Xn[ok], gn[ok], wwl[ok]
    p33, p67 = np.percentile(wwl, [33, 67]); keep = (wwl <= p33) | (wwl >= p67)
    X, g, y = Xn[keep], gn[keep], (wwl[keep] >= p67).astype(int)
    return evaluate(X, y, g)


def run_gcn():
    from pipeline_lengkap.pipeline_final_gcn import prepare, evaluate
    X, g, y = prepare()
    return evaluate(X, y, g)


def run_glct_cnn():
    from pipeline_lengkap.pipeline_final_glct_cnn import prepare, evaluate_full
    H1, g, y = prepare()
    return evaluate_full(H1, y, g)


def run_attnbilstm():
    from pipeline_lengkap.pipeline_final_attnbilstm import prepare, evaluate_full
    X, g, y = prepare()
    return evaluate_full(X, y, g)


def main():
    out = {}
    for name, fn in [("gradboost", run_gradboost), ("gcn", run_gcn),
                     ("glct_cnn", run_glct_cnn), ("attnbilstm", run_attnbilstm)]:
        t = time.time()
        print(f"[{name}] mulai ...", flush=True)
        AL, AP, PR, fa = fn()
        AL, AP, PR, fa = np.array(AL), np.array(AP), np.array(PR), np.array(fa)
        acc = float((AL == AP).mean())
        print(f"[{name}] selesai {time.time()-t:.0f}s -> akurasi={acc:.3f} "
              f"per-fold={np.round(fa,3)}", flush=True)
        out[f"{name}_AL"] = AL; out[f"{name}_AP"] = AP
        out[f"{name}_PR"] = PR; out[f"{name}_fa"] = fa
    np.savez_compressed(OUT, **out)
    print("->", OUT)


if __name__ == "__main__":
    main()
