"""Exp 1 - preprocessing ablation (region x baseline x derivative x normalisation)."""
import sys, time, itertools; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from evalkit import *
from preprocess import SpectraPreproc
from models import PLSDA, _tuned
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

t = get_task(); y = t.y
splits = repeated_cv_splits(y, n_splits=5, n_repeats=5, strat=None)
# stratify CV by class x family x black for stability
key = pd.factorize(strat_key(t.meta))[0]
splits = repeated_cv_splits(y, n_splits=5, n_repeats=5, strat=key)

regions = ['full', 'fp+ch', 'fingerprint', 'ch', 'narrow']
prep = {
 'raw(no norm)':      dict(baseline=None, smooth=0,  deriv=0, norm=None),
 'SNV':               dict(baseline=None, smooth=0,  deriv=0, norm='snv'),
 'max':               dict(baseline=None, smooth=0,  deriv=0, norm='max'),
 'poly+SNV':          dict(baseline='poly', smooth=0, deriv=0, norm='snv'),
 'ALS+SNV':           dict(baseline='als', smooth=0, deriv=0, norm='snv'),
 'SG1(11)+SNV':       dict(baseline=None, smooth=11, deriv=1, norm='snv'),
 'SG2(15)+SNV':       dict(baseline=None, smooth=15, deriv=2, norm='snv'),
 'SNV+SG1(11)':       dict(baseline=None, smooth=11, deriv=1, norm='snv'),  # same impl order (deriv then snv)
}
prep.pop('SNV+SG1(11)')
models = {
 'logreg': lambda: Pipeline([('sc', StandardScaler()), ('clf', _tuned(LogisticRegression(max_iter=2000), {'C': [1e-3, 1e-2, 1e-1, 1]}))]),
 'plsda':  lambda: Pipeline([('sc', StandardScaler(with_std=False)), ('clf', _tuned(PLSDA(), {'n_components': [2, 4, 6, 8, 12]}))]),
 'rbfsvm': lambda: Pipeline([('clf', _tuned(SVC(kernel='rbf', probability=True, random_state=0), {'C': [1, 10, 100], 'gamma': ['scale', 1e-3]}))]),
}
rows = []
Xs = t.X[:, :6845]
t0 = time.time()
for reg, (pn, kw) in itertools.product(regions, prep.items()):
    if kw['baseline'] == 'als' and reg == 'full':
        continue  # ALS on 3500 pts x 178 spectra is slow & ablation-irrelevant
    Z = SpectraPreproc(region=reg, **kw).fit(Xs).transform(Xs)
    for mn, mk in models.items():
        P = oof_predict(mk(), Z, y, splits)
        s = summarise_oof(P, y, t.black); s.update(region=reg, prep=pn, model=mn, n_feat=Z.shape[1])
        rows.append(s)
    print(f"{reg:12s} {pn:14s} done ({time.time()-t0:.0f}s) last acc={s['acc']:.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(RES / 'tables' / 'exp1_preprocessing.csv', index=False)
print(df.sort_values(['acc', 'black_acc'], ascending=False).head(15)[['region', 'prep', 'model', 'n_feat', 'acc', 'acc_sd', 'auc', 'black_acc', 'errors']].to_string())
