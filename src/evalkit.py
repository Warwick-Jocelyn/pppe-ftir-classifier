"""Task definition, splits, metrics and cross-validation helpers shared by all experiments.

Positive class = PP (y=1), negative = PE (y=0).
Feature matrix columns: [0:6845] raw absorbance spectrum, [6845:6848] colorimeter L*, a*, b*.
Only flakes with a clear verified label (PE / PP) are used for training and scoring; `PE-dominant`
(3) and `Uncertain` (5, the white Bopp shred - Polymerflow=PVC / OpenSpecy=Mineral) are kept aside and
only reported as a case study.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

from data import ROOT, build_dataset
from preprocess import NATIVE_WN

N_SPEC = len(NATIVE_WN)
RES = ROOT / "results"
SEED = 42


class Task:
    pass


def get_task() -> Task:
    meta, wn, X = build_dataset()
    assert np.allclose(wn, NATIVE_WN, atol=1e-3), "native grid mismatch"
    t = Task()
    ok = meta["label"].isin(["PE", "PP"]).values
    t.meta_all, t.X_all_spec = meta, X
    t.aside = meta[~ok].reset_index(drop=True)
    t.X_aside = np.hstack([X[~ok], meta.loc[~ok, ["L", "a", "b"]].values])
    t.meta = meta[ok].reset_index(drop=True)
    t.X = np.hstack([X[ok], t.meta[["L", "a", "b"]].values])
    t.y = (t.meta["label"] == "PP").astype(int).values
    t.black = t.meta["is_black"].values
    t.source = t.meta["source"].values
    return t


def strat_key(meta: pd.DataFrame) -> np.ndarray:
    return (meta["label"] + "|" + meta["family"] + "|" + np.where(meta["is_black"], "blk", "col")).values


def holdout_split(task: Task, test_size=0.25, seed=SEED):
    """Fixed dev/test split stratified by class x data family x black. Saved for reproducibility."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev, test = next(sss.split(task.X, strat_key(task.meta)))
    out = RES / "splits"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": task.meta["sample_id"], "split": np.where(np.isin(np.arange(len(task.y)), test), "test", "dev")}
                 ).to_csv(out / "holdout_split.csv", index=False)
    return dev, test


def repeated_cv_splits(y, idx=None, n_splits=5, n_repeats=5, seed=SEED, strat=None):
    """List of (repeat, train_idx, test_idx) using global indices."""
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    s = (y if strat is None else strat)[idx]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(k // n_splits, idx[tr], idx[te]) for k, (tr, te) in enumerate(rskf.split(idx, s))]


def loso_splits(task: Task, groups: str = "source"):
    """Leave-one-source-out (each batch / plate set held out in turn)."""
    g = task.source
    return [(0, np.where(g != s)[0], np.where(g == s)[0]) for s in np.unique(g)]


def metrics_from_proba(y, p, thr=0.5) -> dict:
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else np.nan,
        "f1_pp": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else np.nan,
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "errors": int((pred != y).sum()),
    }
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def _fit_predict(est, X, y, tr, te, rep):
    m = clone(est)
    m.fit(X[tr], y[tr])
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X[te])[:, 1]
    else:
        d = m.decision_function(X[te])
        p = 1 / (1 + np.exp(-d))
    return rep, te, p


def oof_predict(est, X, y, splits, n_jobs=-1):
    """Out-of-fold P(PP). Returns array (n_repeats, n) with NaN where a sample was not tested."""
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(est, X, y, tr, te, rep) for rep, tr, te in splits)
    n_rep = max(r for r, _, _ in splits) + 1
    P = np.full((n_rep, len(y)), np.nan)
    for rep, te, p in res:
        P[rep, te] = p
    return P


def summarise_oof(P, y, black=None) -> dict:
    """Per-repeat pooled metrics -> mean / sd across repeats (+ black subset)."""
    rows, rows_b, rows_nb = [], [], []
    for r in range(P.shape[0]):
        m = ~np.isnan(P[r])
        rows.append(metrics_from_proba(y[m], P[r, m]))
        if black is not None:
            mb, mn = m & black, m & ~black
            if mb.any():
                rows_b.append(metrics_from_proba(y[mb], P[r, mb]))
            if mn.any():
                rows_nb.append(metrics_from_proba(y[mn], P[r, mn]))
    def agg(rows, pref):
        d = pd.DataFrame(rows)
        o = {}
        for c in ["acc", "bal_acc", "f1_pp", "mcc", "auc", "errors"]:
            o[f"{pref}{c}"] = d[c].mean()
            o[f"{pref}{c}_sd"] = d[c].std(ddof=0)
        return o
    out = agg(rows, "")
    if rows_b:
        out.update({k: v for k, v in agg(rows_b, "black_").items() if k.startswith("black_acc") or k.startswith("black_errors")})
    if rows_nb:
        out.update({k: v for k, v in agg(rows_nb, "nonblack_").items() if k.startswith("nonblack_acc") or k.startswith("nonblack_errors")})
    return out


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
