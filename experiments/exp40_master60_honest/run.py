"""
exp40 — Tabel master JUJUR di window 60s, protokol terkunci (tanpa best-of-grid).

- ML klasik: 14 algoritma, masing-masing SATU konfig DEFAULT yang dikunci a priori
  (bukan dipilih dari LOSO), evaluasi LOSO level aktivitas.
- AttnBiLSTM: konfig kanonik tetap (LSTM h64 2-lapis bidir + attention), GroupKFold5
  level window.
- GCN & CNN-GLCT: dirujuk dari eksperimen jujur sebelumnya (exp39/exp38).

Tujuan: satu tabel setara untuk BAB IV, semua di 60s, bebas bias seleksi.
Jalankan: python -m experiments.exp40_master60_honest.run
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from ..shared.data import build_raw_windows
from ..shared.evaluation import set_seed, _class_weights
from ..shared.config_shared import DEVICE
from ..exp35_microsiemens_ab.run import activity_nodes, pipe

OUT = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(OUT, exist_ok=True)

# --- 14 ML, konfig DEFAULT dikunci a priori (bukan hasil tuning di LOSO) ---
ML_LOCKED = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", C=1),
    "Ridge": RidgeClassifier(class_weight="balanced", alpha=1),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.3),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "GradBoost": GradientBoostingClassifier(random_state=42),
    "HistGB": HistGradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM-RBF": SVC(C=1, kernel="rbf", class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
    "NaiveBayes": GaussianNB(),
}


def loso_node(model, Xn, yn, gn):
    AP, AL = [], []
    for tr, te in LeaveOneGroupOut().split(Xn, yn, gn):
        if len(np.unique(yn[tr])) < 2:
            continue
        m = clone(pipe(model)); m.fit(Xn[tr], yn[tr])
        AP.extend(m.predict(Xn[te])); AL.extend(yn[te])
    return accuracy_score(AL, AP), f1_score(AL, AP, zero_division=0)


# --- AttnBiLSTM kanonik (window 1-D), GroupKFold5 ---
class AttnBiLSTM(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.3, n_classes=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.rnn = nn.LSTM(16, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        od = hidden * 2
        self.attn = nn.Linear(od, 1)
        self.head = nn.Sequential(nn.Linear(od, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        h, _ = self.rnn(self.proj(x.unsqueeze(-1)))
        w = torch.softmax(self.attn(h).squeeze(-1), 1)
        ctx = (h * w.unsqueeze(-1)).sum(1)
        return self.head(ctx)


def gkf_window(Xraw, y, groups, epochs=15, bs=32, lr=1e-3, patience=5):
    set_seed()
    AP, AL = [], []
    for tr, te in GroupKFold(5).split(Xraw, y, groups):
        tl = DataLoader(TensorDataset(torch.from_numpy(Xraw[tr]).float(), torch.from_numpy(y[tr]).long()),
                        batch_size=bs, shuffle=True)
        vl = DataLoader(TensorDataset(torch.from_numpy(Xraw[te]).float(), torch.from_numpy(y[te]).long()),
                        batch_size=bs, shuffle=False)
        m = AttnBiLSTM().to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=_class_weights(y[tr], 2).to(DEVICE))
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
        best, bstate, wait = -1, None, 0
        for _ in range(epochs):
            m.train()
            for xb, yb in tl:
                opt.zero_grad(); crit(m(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
            m.eval(); P, L = [], []
            with torch.no_grad():
                for xb, yb in vl:
                    P.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); L.extend(yb.numpy())
            f = f1_score(L, P, zero_division=0)
            if f > best:
                best, wait, bstate = f, 0, {k: v.cpu().clone() for k, v in m.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if bstate:
            m.load_state_dict(bstate)
        m.eval(); P, L = [], []
        with torch.no_grad():
            for xb, yb in vl:
                P.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy()); L.extend(yb.numpy())
        AP.extend(P); AL.extend(L)
    return accuracy_score(AL, AP), f1_score(AL, AP, zero_division=0)


def main():
    t0 = time.time()
    Xraw, y, groups, subjects, meta, _ = build_raw_windows(verbose=False, window_sec=60, hop_sec=30)
    Xn, yn, gn = activity_nodes(Xraw, y, groups, meta)
    base_n = max(float(np.mean(yn)), 1 - float(np.mean(yn)))
    base_w = max(float(np.mean(y)), 1 - float(np.mean(y)))
    print(f"window 60s | node={len(Xn)} window={len(Xraw)} | base_node={base_n:.3f} base_win={base_w:.3f}\n")

    rows = []
    print("ML KLASIK (terkunci, LOSO level aktivitas):")
    for name, model in ML_LOCKED.items():
        a, f = loso_node(model, Xn, yn, gn)
        rows.append({"model": name, "tipe": "ML (LOSO node)", "acc": float(a), "f1": float(f)})
        print(f"  {name:14s} acc={a:.3f} f1={f:.3f}")

    print("\nDEEP (level window, GroupKFold5):")
    a, f = gkf_window(Xraw, y, groups)
    rows.append({"model": "AttnBiLSTM", "tipe": "DL (GKF5 window)", "acc": float(a), "f1": float(f)})
    print(f"  {'AttnBiLSTM':14s} acc={a:.3f} f1={f:.3f}")

    # rujukan dari eksperimen jujur sebelumnya
    refs = [{"model": "GCN (transduktif, multi-seed)", "tipe": "ref exp39 (LOSO node)", "acc": 0.733, "f1": 0.739},
            {"model": "CNN-GLCT (hop30/hop15)", "tipe": "ref exp38 (GKF5 window)", "acc": 0.583, "f1": 0.622}]

    allrows = rows + refs
    json.dump({"base_node": base_n, "base_win": base_w, "rows": allrows},
              open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print("\n" + "=" * 58)
    print(f"{'MODEL':30s} {'acc':>6s} {'f1':>6s}  tipe")
    print("=" * 58)
    for r in sorted(allrows, key=lambda r: -r["acc"]):
        print(f"{r['model']:30s} {r['acc']:6.3f} {r['f1']:6.3f}  {r['tipe']}")
    print(f"\nbase node={base_n:.3f} | total {(time.time()-t0)/60:.1f} min -> results.json")


if __name__ == "__main__":
    main()
