"""Exp 5 - robustness of trained models to realistic acquisition degradations.

Models are trained on CLEAN training folds; the held-out fold's spectra are degraded:
  noise       white noise, sigma = level x spectrum peak
  baseline    random linear tilt + offset, amplitude = level x spectrum peak
  attenuation signal gain (poor contact / dark pigment) with instrument noise kept constant
  blur        Gaussian broadening (lower optical resolution), FWHM in cm-1
  shift       random wavenumber shift (calibration drift), +-level cm-1
Colour columns are never perturbed (separate instrument) - this is where fusion could help.
"""
import sys, warnings, pickle, time; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from scipy.ndimage import gaussian_filter1d
from evalkit import *
from models import *
from deep import TorchSpectraClassifier
from preprocess import NATIVE_WN

t = get_task(); y = t.y
Xs = t.X[:, :N_SPEC]
STEP = float(np.mean(np.diff(NATIVE_WN)))
# measured instrument noise: std of detrended high-wavenumber region with no chemical bands
w = (NATIVE_WN > 5500) & (NATIVE_WN < 6800)
SIG0 = float(np.median([np.std(np.diff(s[w])) / np.sqrt(2) for s in Xs]))
print(f'estimated instrument noise sigma0 = {SIG0:.5f} (peak absorbance median {np.median(Xs.max(1)):.3f})')

GRID = {
    'noise':       [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    'baseline':    [0, 0.02, 0.05, 0.1, 0.2, 0.5],
    'attenuation': [1.0, 0.5, 0.25, 0.1, 0.05, 0.02],
    'blur':        [0, 4, 8, 16, 32],
    'shift':       [0, 1, 2, 4, 8],
}

def perturb(S, kind, lvl, rng):
    S = S.copy(); n, L = S.shape
    pk = S.max(1, keepdims=True)
    if kind == 'noise':
        S += rng.normal(0, 1, S.shape) * lvl * pk
    elif kind == 'baseline':
        tt = np.linspace(-1, 1, L)[None]
        S += pk * lvl * (rng.choice([-1, 1], (n, 1)) * tt + rng.uniform(-1, 1, (n, 1)))
    elif kind == 'attenuation':
        S = S * lvl + rng.normal(0, SIG0, S.shape)
    elif kind == 'blur':
        if lvl > 0:
            S = gaussian_filter1d(S, sigma=lvl / 2.355 / STEP, axis=1)
    elif kind == 'shift':
        if lvl > 0:
            sh = rng.choice([-1, 1], n) * lvl
            S = np.vstack([np.interp(NATIVE_WN, NATIVE_WN + d, s) for s, d in zip(S, sh)])
    return S

def job(est, tr, te, rep, seed):
    m = clone(est).fit(t.X[tr], y[tr])
    rng = np.random.default_rng(seed)
    out = {}
    for kind, lv in GRID.items():
        for l in lv:
            Xt = t.X[te].copy()
            Xt[:, :N_SPEC] = perturb(Xt[:, :N_SPEC], kind, l, rng)
            out[(kind, l)] = m.predict_proba(Xt)[:, 1]
    return rep, te, out

ALL_ZOO = {
    'logreg': make_classical('logreg'), 'plsda': make_classical('plsda'), 'rbfsvm': make_classical('rbfsvm'),
    'extratrees': make_classical('extratrees'), 'lightgbm': make_classical('lightgbm'),
    'cnn1d (aug)': TorchSpectraClassifier(arch='cnn1d', epochs=60), 'resnet1d (aug)': TorchSpectraClassifier(arch='resnet1d', epochs=60),
    'transformer (aug)': TorchSpectraClassifier(arch='transformer', epochs=60),
    'colour-only rf': make_colour('rf'),
    'late fusion (logreg+Lab)': LateFusion(ftir=make_classical('logreg')),
    'early fusion (w=3)': EarlyFusion(colour_weight=3.0),
}
import os
if os.environ.get('CANDIDATES') == 'final':
    from models import AugSVM, LateFusion
    ALL_ZOO = {'AugSVM (deployed, FTIR only)': AugSVM(), 'AugSVM + colour (late fusion)': LateFusion(ftir=AugSVM(), inner=3),
               'colour-only rf': make_colour('rf')}
    OUT_NAME = 'exp8b_final_robustness.csv'
elif os.environ.get('CANDIDATES'):
    from models import AugmentedClassifier, SoftVote
    ALL_ZOO = {
        'rbfsvm': make_classical('rbfsvm'),
        'logreg': make_classical('logreg'),
        'AUG rbfsvm': AugmentedClassifier(make_classical('rbfsvm')),
        'AUG logreg': AugmentedClassifier(make_classical('logreg')),
        'AUG plsda': AugmentedClassifier(make_classical('plsda')),
        'AUG extratrees': AugmentedClassifier(make_classical('extratrees')),
        'ENS logreg+rbfsvm+cnn1d': SoftVote([make_classical('logreg'), make_classical('rbfsvm'), TorchSpectraClassifier(arch='cnn1d', epochs=60)]),
        'ENS AUGrbfsvm+AUGlogreg+cnn1d': SoftVote([AugmentedClassifier(make_classical('rbfsvm')), AugmentedClassifier(make_classical('logreg')), TorchSpectraClassifier(arch='cnn1d', epochs=60)]),
        'ENS AUGrbfsvm+cnn1d': SoftVote([AugmentedClassifier(make_classical('rbfsvm')), TorchSpectraClassifier(arch='cnn1d', epochs=60)]),
    }
    OUT_NAME = 'exp8_candidates_robustness.csv'
else:
    OUT_NAME = 'exp5_robustness.csv'
zoo = ALL_ZOO
key = pd.factorize(strat_key(t.meta))[0]
splits = repeated_cv_splits(y, n_splits=5, n_repeats=2, strat=key)
rows = []
t0 = time.time()
for name, est in zoo.items():
    res = Parallel(n_jobs=30)(delayed(job)(est, tr, te, rep, 100 + k) for k, (rep, tr, te) in enumerate(splits))
    for (kind, l) in [(k, l) for k, ls in GRID.items() for l in ls]:
        P = np.full((2, len(y)), np.nan)
        for rep, te, out in res: P[rep, te] = out[(kind, l)]
        s = summarise_oof(P, y, t.black); s.update(model=name, kind=kind, level=l); rows.append(s)
    print(f'{name:28s} done ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(RES / 'tables' / OUT_NAME, index=False)
save_json({'sigma0': SIG0, 'grid': GRID}, RES / 'tables' / 'exp5_meta.json')
