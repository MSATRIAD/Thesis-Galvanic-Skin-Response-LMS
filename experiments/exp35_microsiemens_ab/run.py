"""
exp35 — A/B: dataset ADC mentah vs dataset KONDUKTANSI (microsiemens, uS).

Tujuan: melihat pengaruh konversi Grove (gsrRaw ADC -> uS; nonlinear + polaritas
benar) ke kinerja SELURUH keluarga model, pada protokol subject-independent yang
IDENTIK:
  - TABULAR : fitur EXTENDED -> agregasi per-aktivitas -> 14 model ML (LOSO).
  - SPATIAL : heatmap GLCT -> CNN (GroupKFold 5).
  - GRAPH   : kNN graph fitur -> GCN transduktif (LOOCV per-subjek).

Dataset uS dimuat lewat build_raw_windows(data_dir=..., gsr_column='gsr_uS',
data_tag='uS_') -> cache terpisah, tak menimpa cache ADC.

Jalankan: python -m experiments.exp35_microsiemens_ab.run
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, time
import numpy as np
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn

from ..shared.data import build_raw_windows
from ..shared.config_shared import FS, PROJECT_ROOT
from ..shared.evaluation import cross_validate
from ..exp02_feature_ablation import feature_sets as FS_
from ..exp01_glct_cnn.representation import build_heatmaps
from ..exp01_glct_cnn.model import CognitiveLoadCNN
from ..exp01_glct_cnn import config as C1

OUT = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(OUT, exist_ok=True)
US_DIR = str(PROJECT_ROOT / "Data responden uS")


# --------------------------------------------------------------- util agregasi
def activity_nodes(Xraw, y, groups, meta):
    """Fitur EXTENDED per window -> rata-rata per aktivitas (subjek x task)."""
    X = np.nan_to_num(np.array([FS_.extended(w, FS) for w in Xraw]))
    by = defaultdict(list)
    for i, m in enumerate(meta):
        by[m].append(i)
    Xn, yn, gn = [], [], []
    for m, idx in by.items():
        idx = np.array(idx)
        Xn.append(X[idx].mean(0)); yn.append(y[idx[0]]); gn.append(groups[idx[0]])
    return np.array(Xn), np.array(yn), np.array(gn)


# --------------------------------------------------------------- TABULAR (ML)
def pipe(model, k=12):
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_classif, k=k)), ("clf", model)])


GRID = {
    "LogReg": [pipe(LogisticRegression(max_iter=1000, class_weight="balanced", C=c)) for c in [0.1, 1, 10]],
    "Ridge": [pipe(RidgeClassifier(class_weight="balanced", alpha=a)) for a in [0.5, 1, 5]],
    "LDA": [pipe(LinearDiscriminantAnalysis())],
    "QDA": [pipe(QuadraticDiscriminantAnalysis(reg_param=r), k=8) for r in [0.3, 0.6]],
    "RF": [pipe(RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, class_weight="balanced"))
           for n in [200, 400] for d in [None, 5]],
    "ExtraTrees": [pipe(ExtraTreesClassifier(n_estimators=n, random_state=42, class_weight="balanced")) for n in [200, 400]],
    "GradBoost": [pipe(GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42))
                  for lr in [0.05, 0.1] for n in [100, 200]],
    "HistGB": [pipe(HistGradientBoostingClassifier(learning_rate=lr, max_iter=n, random_state=42))
               for lr in [0.05, 0.1] for n in [100, 200]],
    "AdaBoost": [pipe(AdaBoostClassifier(n_estimators=n, random_state=42)) for n in [100, 200]],
    "SVM": [pipe(SVC(C=c, kernel=k, class_weight="balanced", random_state=42)) for c in [0.1, 1, 10] for k in ["rbf", "linear"]],
    "KNN": [pipe(KNeighborsClassifier(n_neighbors=k)) for k in [3, 5, 7, 9]],
    "DecisionTree": [pipe(DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced")) for d in [3, 5, None]],
    "MLP": [pipe(MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=800, random_state=42))
            for h in [(32,), (64, 32)] for a in [1e-4, 1e-2]],
    "NaiveBayes": [pipe(GaussianNB())],
}


def _loso_pipe(p, Xn, yn, gn):
    AP, AL = [], []
    try:
        for tr, te in LeaveOneGroupOut().split(Xn, yn, gn):
            if len(np.unique(yn[tr])) < 2:
                continue
            m = clone(p); m.fit(Xn[tr], yn[tr]); AP.extend(m.predict(Xn[te])); AL.extend(yn[te])
    except Exception:
        return 0.0, 0.0
    return accuracy_score(AL, AP), f1_score(AL, AP, zero_division=0)


def run_tabular(Xn, yn, gn):
    res = {}
    for name, cfgs in GRID.items():
        a, f, _ = max((_loso_pipe(c, Xn, yn, gn) + (c,) for c in cfgs), key=lambda t: t[0])
        res[name] = {"acc": float(a), "f1": float(f)}
    best = max(res.items(), key=lambda kv: kv[1]["acc"])
    return res, {"model": best[0], **best[1]}


# --------------------------------------------------------------- GRAPH (GCN)
def knn_adj(F, k=8):
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T; np.fill_diagonal(S, -1)
    A = np.zeros_like(S)
    for i in range(len(S)):
        A[i, np.argsort(S[i])[::-1][:k]] = 1
    return np.maximum(A, A.T)


def norm_adj(A):
    A = A + np.eye(len(A))
    d = A.sum(1); dinv = np.power(d, -0.5); dinv[np.isinf(dinv)] = 0
    return (dinv[:, None] * A) * dinv[None, :]


class GCN(nn.Module):
    def __init__(self, fin, h=32, nc=2, p=0.5):
        super().__init__()
        self.w1 = nn.Linear(fin, h); self.w2 = nn.Linear(h, nc)
        self.drop = nn.Dropout(p); self.act = nn.ReLU()

    def forward(self, Ah, X):
        h = self.act(Ah @ self.w1(X)); h = self.drop(h)
        return self.w2(Ah @ h)


def run_graph(Xn, yn, gn, k=8, epochs=150):
    torch.manual_seed(42); np.random.seed(42)
    sc = StandardScaler().fit(Xn); Xs = sc.transform(Xn)
    Ah = norm_adj(knn_adj(Xs, k)).astype(np.float32)
    At = torch.tensor(Ah); Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(yn, dtype=torch.long)
    AP, AL = [], []
    for s in np.unique(gn):
        te = gn == s; tr = ~te
        if len(np.unique(yn[tr])) < 2:
            continue
        model = GCN(Xs.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        cls, cnt = np.unique(yn[tr], return_counts=True)
        w = torch.tensor([(tr.sum() / (2 * cnt[list(cls).index(c)])) if c in cls else 1.0
                          for c in [0, 1]], dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w)
        tm = torch.tensor(tr)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss = crit(model(At, Xt)[tm], yt[tm]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(At, Xt).argmax(1).numpy()
        AP.extend(pred[te]); AL.extend(yn[te])
    return {"acc": float(accuracy_score(AL, AP)),
            "f1": float(f1_score(AL, AP, zero_division=0))}


# --------------------------------------------------------------- SPATIAL (CNN)
def run_spatial(Xraw, y, groups, subjects, subj_stats, tag, epochs=15, kfold=5):
    X = build_heatmaps(Xraw, groups=groups, subj_stats=subj_stats, apply_hpf=True, tag=tag)
    res = cross_validate(lambda: CognitiveLoadCNN(int(np.max(y)) + 1), X, y, groups, subjects,
                         n_folds=kfold, epochs=epochs, lr=C1.LEARNING_RATE,
                         weight_decay=C1.WEIGHT_DECAY, batch_size=C1.BATCH_SIZE,
                         patience=C1.EARLY_STOPPING_PATIENCE, verbose=True, progress=False)
    return {"acc": float(res["accuracy"]), "f1": float(res["f1"])}


# --------------------------------------------------------------- driver
def load_variant(which):
    if which == "ADC":
        return build_raw_windows(verbose=False)
    return build_raw_windows(verbose=False, data_dir=US_DIR,
                             gsr_column="gsr_uS", data_tag="uS_")


def main():
    summary = {}
    for which in ["ADC", "uS"]:
        print("\n" + "#" * 70)
        print(f"#  VARIAN DATA: {which}")
        print("#" * 70)
        t0 = time.time()
        Xraw, y, groups, subjects, meta, subj_stats = load_variant(which)
        base = max(float(np.mean(y)), 1 - float(np.mean(y)))
        Xn, yn, gn = activity_nodes(Xraw, y, groups, meta)
        node_base = max(float(np.mean(yn)), 1 - float(np.mean(yn)))
        print(f"  windows={len(Xraw)} | node aktivitas={len(Xn)} | "
              f"base(window)={base:.3f} base(node)={node_base:.3f}")

        print("\n[1/3] TABULAR (14 model ML, LOSO) ...")
        tab_all, tab_best = run_tabular(Xn, yn, gn)
        print(f"      terbaik: {tab_best['model']} acc={tab_best['acc']:.3f} f1={tab_best['f1']:.3f}")

        print("\n[2/3] GRAPH (GCN transduktif, LOOCV) ...")
        gph = run_graph(Xn, yn, gn)
        print(f"      GCN acc={gph['acc']:.3f} f1={gph['f1']:.3f}")

        print("\n[3/3] SPATIAL (GLCT-CNN, GroupKFold 5) ...")
        spa = run_spatial(Xraw, y, groups, subjects, subj_stats, tag=f"ab_{which}")
        print(f"      CNN acc={spa['acc']:.3f} f1={spa['f1']:.3f}")

        summary[which] = {
            "base_node": node_base, "n_nodes": int(len(Xn)), "n_windows": int(len(Xraw)),
            "tabular_best": tab_best, "tabular_all": tab_all,
            "graph_gcn": gph, "spatial_cnn": spa,
            "minutes": round((time.time() - t0) / 60, 1),
        }
        json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    # ---- tabel A/B ----
    print("\n" + "=" * 70)
    print("RINGKASAN A/B  (acc | f1)            ADC            ->   uS")
    print("=" * 70)
    def cell(d): return f"{d['acc']:.3f}/{d['f1']:.3f}"
    rows = [("Tabular (best ML)", "tabular_best"),
            ("Graph (GCN)", "graph_gcn"),
            ("Spatial (CNN)", "spatial_cnn")]
    for label, key in rows:
        a = summary["ADC"][key]; b = summary["uS"][key]
        extra = f"  [{summary['ADC']['tabular_best']['model']} / {summary['uS']['tabular_best']['model']}]" \
                if key == "tabular_best" else ""
        print(f"  {label:20s} {cell(a):14s} -> {cell(b):14s}{extra}")
    print(f"\n  base rate (node): ADC={summary['ADC']['base_node']:.3f}  uS={summary['uS']['base_node']:.3f}")
    print(f"  hasil -> {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
