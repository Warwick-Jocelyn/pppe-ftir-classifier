"""Model zoo: classical chemometrics / ML, colour-only, fusion, and PyTorch 1-D backbones.

Every estimator consumes the raw feature matrix [spectrum (6845) | L* a* b*] so the exact same
object can be cross-validated and deployed. Positive class = PP.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import NATIVE_WN, SpectraPreproc

N_SPEC = len(NATIVE_WN)


# ----------------------------------------------------------------------------- column selectors
class SpecCols(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, :N_SPEC]


class ColourCols(BaseEstimator, TransformerMixin):
    """L*, a*, b* + derived chroma C* and lightness-normalised features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        c = np.asarray(X)[:, N_SPEC:N_SPEC + 3].astype(float)
        L, a, b = c[:, 0], c[:, 1], c[:, 2]
        chroma = np.hypot(a, b)
        return np.column_stack([L, a, b, chroma])


# ----------------------------------------------------------------------------- PLS-DA
class PLSDA(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=8):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        nc = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=nc, scale=False).fit(X, np.asarray(y, float))
        return self

    def decision_function(self, X):
        return self.pls_.predict(X).ravel() - 0.5

    def predict_proba(self, X):
        p = np.clip(self.pls_.predict(X).ravel(), 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def _cv(k=3):
    return StratifiedKFold(k, shuffle=True, random_state=0)


def _tuned(est, grid):
    return GridSearchCV(est, grid, cv=_cv(3), scoring="neg_log_loss" if hasattr(est, "predict_proba") else "accuracy",
                        n_jobs=1, refit=True) if grid else est


# ----------------------------------------------------------------------------- classical zoo
def make_classical(name: str, pp: dict | None = None):
    """FTIR-only classical pipeline. `pp` = SpectraPreproc kwargs."""
    pp = pp or dict(region="fp+ch", baseline=None, smooth=11, deriv=1, norm="snv")
    steps = [("sel", SpecCols()), ("pp", SpectraPreproc(**pp))]
    if name == "logreg":
        clf = _tuned(LogisticRegression(max_iter=2000), {"C": [1e-3, 1e-2, 1e-1, 1, 10]})
        steps += [("sc", StandardScaler()), ("clf", clf)]
    elif name == "linsvm":
        clf = _tuned(SVC(kernel="linear", probability=True, random_state=0), {"C": [1e-3, 1e-2, 1e-1, 1]})
        steps += [("sc", StandardScaler()), ("clf", clf)]
    elif name == "rbfsvm":
        clf = _tuned(SVC(kernel="rbf", probability=True, random_state=0), {"C": [1, 10, 100], "gamma": ["scale", 1e-4, 1e-3]})
        steps += [("clf", clf)]
    elif name == "plsda":
        clf = _tuned(PLSDA(), {"n_components": [2, 3, 4, 6, 8, 12]})
        steps += [("sc", StandardScaler(with_std=False)), ("clf", clf)]
    elif name == "pca_lda":
        steps += [("sc", StandardScaler()), ("pca", PCA(n_components=15, random_state=0)), ("clf", LinearDiscriminantAnalysis())]
    elif name == "knn":
        clf = _tuned(KNeighborsClassifier(metric="cosine", weights="distance"), {"n_neighbors": [1, 3, 5, 9]})
        steps += [("clf", clf)]
    elif name == "rf":
        steps += [("clf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1))]
    elif name == "extratrees":
        steps += [("clf", ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=1))]
    elif name == "xgboost":
        from xgboost import XGBClassifier
        steps += [("clf", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.3,
                                        n_jobs=1, random_state=0, verbosity=0, eval_metric="logloss"))]
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        steps += [("clf", LGBMClassifier(n_estimators=250, num_leaves=8, learning_rate=0.06, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.3, min_child_samples=5, n_jobs=1, random_state=0, verbose=-1))]
    else:
        raise ValueError(name)
    return Pipeline(steps)


CLASSICAL = ["logreg", "linsvm", "rbfsvm", "plsda", "pca_lda", "knn", "rf", "extratrees", "xgboost", "lightgbm"]


# ----------------------------------------------------------------------------- colour only
def make_colour(name: str = "logreg"):
    steps = [("sel", ColourCols()), ("sc", StandardScaler())]
    if name == "logreg":
        steps.append(("clf", _tuned(LogisticRegression(max_iter=2000), {"C": [0.1, 1, 10]})))
    elif name == "rbfsvm":
        steps.append(("clf", _tuned(SVC(kernel="rbf", probability=True, random_state=0), {"C": [1, 10], "gamma": ["scale", 0.1]})))
    elif name == "rf":
        steps.append(("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1)))
    return Pipeline(steps)


# ----------------------------------------------------------------------------- fusion
class EarlyFusion(ClassifierMixin, BaseEstimator):
    """Concatenate preprocessed spectrum with colour features (colour block up-weighted)."""

    def __init__(self, pp=None, colour_weight=5.0, C=0.1):
        self.pp = pp
        self.colour_weight = colour_weight
        self.C = C

    def _feat(self, X, fit=False):
        S = self.pp_.transform(X[:, :N_SPEC])
        Cc = ColourCols().transform(X)
        if fit:
            self.ssp_, self.ssc_ = StandardScaler().fit(S), StandardScaler().fit(Cc)
        return np.hstack([self.ssp_.transform(S), self.colour_weight * self.ssc_.transform(Cc)])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pp_ = SpectraPreproc(**(self.pp or dict(region="fp+ch", smooth=11, deriv=1, norm="snv"))).fit(X[:, :N_SPEC])
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000).fit(self._feat(np.asarray(X), True), y)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._feat(np.asarray(X)))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class LateFusion(ClassifierMixin, BaseEstimator):
    """Stack: out-of-fold FTIR logit + colour features -> small logistic regression."""

    def __init__(self, ftir=None, colour_only_features=True, C=1.0, inner=5):
        self.ftir = ftir
        self.colour_only_features = colour_only_features
        self.C = C
        self.inner = inner

    def _stack(self, X, z):
        feats = [z[:, None]]
        if self.colour_only_features:
            feats.append(self.sc_.transform(ColourCols().transform(X)))
        return np.hstack(feats)

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        base = self.ftir if self.ftir is not None else make_classical("logreg")
        k = min(self.inner, int(np.bincount(y).min()))
        oof = cross_val_predict(clone(base), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=1), method="predict_proba")[:, 1]
        self.base_ = clone(base).fit(X, y)
        self.sc_ = StandardScaler().fit(ColourCols().transform(X))
        self.meta_ = LogisticRegression(C=self.C, max_iter=2000).fit(self._stack(X, self._logit(oof)), y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        z = self._logit(self.base_.predict_proba(X)[:, 1])
        return self.meta_.predict_proba(self._stack(X, z))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ----------------------------------------------------------------------------- ensemble
class SoftVote(ClassifierMixin, BaseEstimator):
    """Average P(PP) of several fitted pipelines (all consume the raw [spectrum | Lab] matrix)."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]
        return self

    def predict_proba(self, X):
        w = np.ones(len(self.fitted_)) if self.weights is None else np.asarray(self.weights, float)
        p = sum(wi * m.predict_proba(X)[:, 1] for wi, m in zip(w, self.fitted_)) / w.sum()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AugmentedClassifier(ClassifierMixin, BaseEstimator):
    """Train a base pipeline on the original flakes + `n_aug` randomly degraded copies of each
    (noise, blur, gain, baseline, +-8 cm-1 shift). Inference is unchanged (no test-time cost)."""

    def __init__(self, base=None, n_aug=6, seed=0):
        self.base = base
        self.n_aug = n_aug
        self.seed = seed

    def fit(self, X, y):
        from deep import augment_strong
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        parts, ys = [X], [y]
        for _ in range(self.n_aug):
            Xa = X.copy()
            Xa[:, :N_SPEC] = augment_strong(X[:, :N_SPEC], rng)
            parts.append(Xa)
            ys.append(y)
        self.classes_ = np.unique(y)
        self.model_ = clone(self.base).fit(np.vstack(parts), np.concatenate(ys))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class AugSVM(ClassifierMixin, BaseEstimator):
    """Deployment model: RBF-SVM trained on original + degraded copies of every flake.

    Hyper-parameters and Platt calibration use GroupKFold with groups = original flake, so augmented
    copies of the same flake never straddle inner train/validation folds (no leakage). Calibration is
    fitted on out-of-fold decision values of the augmented set, i.e. it reflects accuracy on degraded
    spectra too. Input = raw [spectrum | Lab] matrix (Lab ignored)."""

    def __init__(self, pp=None, n_aug=6, seed=0, Cs=(1.0, 10.0, 100.0), gammas=("scale", 1e-4, 1e-3), inner=5):
        self.pp = pp
        self.n_aug = n_aug
        self.seed = seed
        self.Cs = Cs
        self.gammas = gammas
        self.inner = inner

    def _pipe(self, **svc):
        pp = self.pp or dict(region="fp+ch", smooth=11, deriv=1, norm="snv")
        return Pipeline([("sel", SpecCols()), ("pp", SpectraPreproc(**pp)), ("svc", SVC(kernel="rbf", **svc))])

    def fit(self, X, y):
        from deep import augment_strong
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.model_selection import GroupKFold
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        parts, ys, gs = [X], [y], [np.arange(len(y))]
        for _ in range(self.n_aug):
            Xa = X.copy()
            Xa[:, :N_SPEC] = augment_strong(X[:, :N_SPEC], rng)
            parts.append(Xa); ys.append(y); gs.append(np.arange(len(y)))
        XA, yA, gA = np.vstack(parts), np.concatenate(ys), np.concatenate(gs)
        self.classes_ = np.unique(y)
        gcv = GroupKFold(n_splits=min(self.inner, len(y)))
        gs_ = GridSearchCV(self._pipe(), {"svc__C": list(self.Cs), "svc__gamma": list(self.gammas)}, cv=gcv, scoring="roc_auc", n_jobs=1, refit=False)
        gs_.fit(XA, yA, groups=gA)
        self.best_params_ = gs_.best_params_
        pipe = self._pipe(C=self.best_params_["svc__C"], gamma=self.best_params_["svc__gamma"])
        oof = np.zeros(len(yA))
        for tr, te in gcv.split(XA, yA, gA):
            oof[te] = clone(pipe).fit(XA[tr], yA[tr]).decision_function(XA[te])
        self.platt_ = LR(C=1e3).fit(oof[:, None], yA)
        self.model_ = pipe.fit(XA, yA)
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        return self.platt_.predict_proba(self.decision_function(X)[:, None])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
