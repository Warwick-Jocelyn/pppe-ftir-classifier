"""Stateless spectral preprocessing (safe inside CV: nothing is fitted on data).

`SpectraPreproc` is an sklearn transformer taking raw absorbance on the *native* grid
(`NATIVE_WN`, 6845 pts) and applying, in order:
    crop(s) -> [resample/decimate] -> baseline -> smoothing/derivative -> normalisation
The same object is used for training, evaluation and the web service, so what is benchmarked
is exactly what is deployed.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin

# Native OPUS export grid (ascending), reproduced from the data.
NATIVE_WN = np.linspace(399.2136, 6998.77372, 6845)

REGIONS = {
    "full": [(450, 4000)],
    "fingerprint": [(600, 1800)],
    "ch": [(2700, 3100)],
    "fp+ch": [(600, 1800), (2700, 3100)],
    "narrow": [(680, 780), (1340, 1500), (2800, 3000)],  # PE doublet, CH2/CH3 bend, CH stretch
}


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 8) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    DDT = lam * D.dot(D.T)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DDT).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def poly_baseline(x: np.ndarray, y: np.ndarray, order: int = 2, n_iter: int = 20) -> np.ndarray:
    """Iterative modified polynomial fit (Lieber & Mahadevan-Jansen) - fast, deterministic."""
    yy = y.copy()
    xs = (x - x.mean()) / (x.max() - x.min())
    for _ in range(n_iter):
        c = np.polyfit(xs, yy, order)
        base = np.polyval(c, xs)
        yy = np.minimum(yy, base)
    return base


class SpectraPreproc(BaseEstimator, TransformerMixin):
    """Parameters
    region : key of REGIONS (or list of (lo, hi) tuples)
    step   : keep every `step`-th native point (1 = native 0.96 cm-1 spacing)
    baseline : None | 'poly' | 'als' | 'offset' (subtract min)
    smooth : Savitzky-Golay window (odd int, 0 = off)
    deriv  : 0/1/2 SG derivative order (needs smooth>0)
    norm   : None | 'snv' | 'max' | 'area' | 'l2'
    """

    def __init__(self, region="full", step=1, baseline=None, smooth=0, deriv=0, norm="snv", wn=None):
        self.region = region
        self.step = step
        self.baseline = baseline
        self.smooth = smooth
        self.deriv = deriv
        self.norm = norm
        self.wn = wn

    # -- helpers ---------------------------------------------------------------------------
    def _wn(self):
        return NATIVE_WN if self.wn is None else np.asarray(self.wn)

    def _mask(self):
        wn = self._wn()
        regs = REGIONS[self.region] if isinstance(self.region, str) else self.region
        m = np.zeros(len(wn), bool)
        for lo, hi in regs:
            m |= (wn >= lo) & (wn <= hi)
        idx = np.where(m)[0]
        return idx[:: max(1, int(self.step))]

    def fit(self, X, y=None):
        self.idx_ = self._mask()
        self.wn_out_ = self._wn()[self.idx_]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[None]
        wn = self._wn()
        regs = REGIONS[self.region] if isinstance(self.region, str) else self.region
        # process each contiguous region separately (baseline / derivatives must not cross gaps)
        out = []
        for lo, hi in regs:
            m = np.where((wn >= lo) & (wn <= hi))[0]
            seg = X[:, m]
            xs = wn[m]
            if self.baseline == "poly":
                seg = seg - np.vstack([poly_baseline(xs, s) for s in seg])
            elif self.baseline == "als":
                seg = seg - np.vstack([als_baseline(s) for s in seg])
            elif self.baseline == "offset":
                seg = seg - seg.min(1, keepdims=True)
            if self.smooth:
                seg = savgol_filter(seg, self.smooth, 3, deriv=self.deriv, axis=1)
            out.append(seg[:, :: max(1, int(self.step))])
        Z = np.hstack(out)
        if self.norm == "snv":
            Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-12)
        elif self.norm == "max":
            Z = Z / (np.abs(Z).max(1, keepdims=True) + 1e-12)
        elif self.norm == "area":
            Z = Z / (np.abs(Z).sum(1, keepdims=True) + 1e-12)
        elif self.norm == "l2":
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        return Z

    def get_wn_out(self):
        wn = self._wn()
        regs = REGIONS[self.region] if isinstance(self.region, str) else self.region
        return np.hstack([wn[(wn >= lo) & (wn <= hi)][:: max(1, int(self.step))] for lo, hi in regs])


def resample_to_native(wn_in: np.ndarray, a_in: np.ndarray) -> np.ndarray:
    """Interpolate an arbitrary-grid spectrum onto the native grid (needed for web uploads)."""
    order = np.argsort(wn_in)
    wn_in, a_in = np.asarray(wn_in)[order], np.asarray(a_in)[order]
    return np.interp(NATIVE_WN, wn_in, a_in, left=a_in[0], right=a_in[-1])
