"""
Evaluasi BERSAMA: subject-independent CV (LOSO / GroupKFold) untuk model PyTorch.

Dipakai semua eksperimen agar protokol evaluasi identik. Arsitektur cukup
menyediakan `model_fn` (callable -> nn.Module baru) dan input X (N,C,H,W) atau
(N,L) sesuai modelnya.

Mendukung biner & multi-kelas (metrik otomatis 'binary' vs 'macro').
Menampilkan progress per-epoch + ETA per-fold supaya tahu sisa waktu.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from .config_shared import DEVICE, RANDOM_SEED

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed); torch.manual_seed(seed)


def _class_weights(y_tr, n_classes):
    classes, counts = np.unique(y_tr, return_counts=True)
    w = len(y_tr) / (len(classes) * counts)
    full = np.ones(n_classes, dtype=np.float32)
    for c, wi in zip(classes, w):
        full[int(c)] = wi
    return torch.tensor(full, dtype=torch.float32)


def _to_tensor(X):
    t = torch.from_numpy(X).float()
    if t.ndim == 3:        # (N,H,W) -> (N,1,H,W)
        t = t.unsqueeze(1)
    return t


def _infer(model, loader):
    model.eval(); P, L = [], []
    with torch.no_grad():
        for xb, yb in loader:
            P.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy()); L.extend(yb.numpy())
    return np.array(P), np.array(L)


def _train_fold(model_fn, X, y, tr, te, epochs, lr, weight_decay, batch_size,
                patience, n_classes, avg, epoch_desc=None):
    Xtr, Xte = _to_tensor(X[tr]), _to_tensor(X[te])
    ytr, yte = torch.from_numpy(y[tr]).long(), torch.from_numpy(y[te]).long()
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    model = model_fn().to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=_class_weights(y[tr], n_classes).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs_iter = range(epochs)
    if epoch_desc is not None and tqdm is not None:
        epochs_iter = tqdm(epochs_iter, desc=epoch_desc, leave=False)

    best_f1, best_state, wait = -1.0, None, 0
    for _ in epochs_iter:
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
        p, l = _infer(model, vl)
        f1 = f1_score(l, p, average=avg, zero_division=0)
        if hasattr(epochs_iter, "set_postfix"):
            epochs_iter.set_postfix(val_f1=f"{f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state, wait = f1, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return _infer(model, vl)


def cross_validate(model_fn, X, y, groups, subjects, n_folds=None, epochs=25,
                   lr=1e-3, weight_decay=1e-4, batch_size=32, patience=6,
                   verbose=True, progress=True):
    """
    n_folds=None -> Leave-One-Subject-Out ; n_folds=k -> GroupKFold(k).
    model_fn: callable -> nn.Module baru tiap fold (jumlah output harus = n_kelas).
    progress: tampilkan progress per-epoch (tqdm) + ETA per-fold.
    """
    set_seed()
    n_classes = int(np.max(y)) + 1
    avg = "binary" if n_classes == 2 else "macro"
    splitter = LeaveOneGroupOut() if n_folds is None else GroupKFold(n_splits=n_folds)
    scheme = "LOSO" if n_folds is None else f"GroupKFold({n_folds})"
    splits = list(splitter.split(X, y, groups))
    n_splits = len(splits)
    if verbose:
        print(f"\n{'='*64}\nEVALUASI {scheme} | device={DEVICE} | "
              f"{n_classes} kelas | metrik='{avg}'\n{'='*64}")

    AP, AL, rows = [], [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            continue
        tf = time.time()
        desc = f"fold {fold+1}/{n_splits}" if (progress and verbose) else None
        p, l = _train_fold(model_fn, X, y, tr, te, epochs, lr, weight_decay,
                           batch_size, patience, n_classes, avg, epoch_desc=desc)
        AP.extend(p); AL.extend(l)
        ts = sorted({subjects[g] for g in np.unique(groups[te])})
        a = accuracy_score(l, p); f = f1_score(l, p, average=avg, zero_division=0)
        rows.append((",".join(ts), len(te), a, f))
        if verbose:
            done = fold + 1; el = time.time() - t0; eta = el / done * (n_splits - done)
            tstr = (",".join(ts)[:20] + "..") if len(",".join(ts)) > 22 else ",".join(ts)
            print(f"  fold {done:2d}/{n_splits} | test={tstr:22s} | n={len(te):4d} "
                  f"| acc={a:.3f} f1={f:.3f} | {time.time()-tf:4.0f}s | ETA ~{eta/60:4.1f} min")

    AP, AL = np.array(AP), np.array(AL)
    res = {
        "scheme": scheme, "n_classes": n_classes, "metric_avg": avg,
        "accuracy": accuracy_score(AL, AP),
        "precision": precision_score(AL, AP, average=avg, zero_division=0),
        "recall": recall_score(AL, AP, average=avg, zero_division=0),
        "f1": f1_score(AL, AP, average=avg, zero_division=0),
        "confusion_matrix": confusion_matrix(AL, AP).tolist(),
        "fold_acc_mean": float(np.mean([r[2] for r in rows])),
        "fold_acc_std": float(np.std([r[2] for r in rows])),
        "fold_f1_mean": float(np.mean([r[3] for r in rows])),
        "fold_f1_std": float(np.std([r[3] for r in rows])),
        "fold_rows": [{"test": r[0], "n": r[1], "acc": r[2], "f1": r[3]} for r in rows],
    }
    if verbose:
        print(f"\n{'='*64}\nHASIL {scheme} (pooled, {n_classes} kelas)\n{'='*64}")
        for k in ("accuracy", "precision", "recall", "f1"):
            print(f"  {k:10s}: {res[k]:.4f}")
        print(f"  per-fold acc: {res['fold_acc_mean']:.3f} ± {res['fold_acc_std']:.3f}")
        print(f"  per-fold f1 : {res['fold_f1_mean']:.3f} ± {res['fold_f1_std']:.3f}")
        print(f"  confusion   : {res['confusion_matrix']}")
        print(f"  total waktu : {(time.time()-t0)/60:.1f} min")
    return res
