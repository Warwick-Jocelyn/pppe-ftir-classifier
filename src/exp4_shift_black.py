"""Exp 4 - distribution-shift and black-polymer experiments.

 A. family shift      : train Impact -> test Circular 11, and vice versa
 B. zero-shot black   : train on NON-black flakes only, test on black flakes
 C. black-only CV     : repeated CV inside the 22 black flakes (FTIR vs colour vs fusion)
 D. source-confound check for colour: how well does *source/batch* alone explain the label?
"""
import sys, warnings, pickle; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from evalkit import *
from models import *

t = get_task(); y = t.y
fam = t.meta.family.values
zoo = {
    'FTIR logreg': make_classical('logreg'), 'FTIR plsda': make_classical('plsda'), 'FTIR rbfsvm': make_classical('rbfsvm'),
    'FTIR extratrees': make_classical('extratrees'),
    'Colour rf': make_colour('rf'), 'Colour logreg': make_colour('logreg'),
    'Early fusion (w=3)': EarlyFusion(colour_weight=3.0), 'Late fusion (logreg+Lab)': LateFusion(ftir=make_classical('logreg')),
}
rows = []

def record(exp, name, tr, te, extra=None):
    m = zoo[name]
    P = oof_predict(m, t.X, y, [(0, tr, te)], n_jobs=1)[0]
    ok = ~np.isnan(P)
    s = metrics_from_proba(y[ok], P[ok]); s.update(exp=exp, model=name, n_train=len(tr), n_test=int(ok.sum()), n_test_pp=int(y[ok].sum()))
    if extra: s.update(extra)
    rows.append(s); return P

# A. family shift
for a, b in [('Impact', 'Circular11'), ('Circular11', 'Impact')]:
    tr, te = np.where(fam == a)[0], np.where(fam == b)[0]
    for n in zoo: record(f'A: train {a} -> test {b}', n, tr, te)
# B. zero-shot black
tr, te = np.where(~t.black)[0], np.where(t.black)[0]
for n in zoo: record('B: train non-black -> test black', n, tr, te)
# B2. non-black Circular11 -> black Circular11 (all black PE; single class: reports accuracy only)
# C. black-only repeated CV
bl = np.where(t.black)[0]
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=0)
sp = [(k // 3, bl[tr], bl[te]) for k, (tr, te) in enumerate(rskf.split(bl, y[bl]))]
for n, m in zoo.items():
    P = oof_predict(m, t.X, y, sp, n_jobs=-1)
    s = summarise_oof(P[:, :], y, None); s.update(exp='C: CV within black flakes only (n=22: 17 PE, 5 PP)', model=n, n_test=len(bl))
    rows.append(s)
df = pd.DataFrame(rows)
df.to_csv(RES / 'tables' / 'exp4_shift_black.csv', index=False)
pd.set_option('display.width', 250)
print(df[['exp', 'model', 'n_train', 'n_test', 'acc', 'bal_acc', 'auc', 'errors']].round(3).to_string())

# D. confound: source-only predictor (majority class per source, evaluated leave-one-flake-out) and
#    cross-tab of colour name vs label
src = t.source
maj = pd.Series(y).groupby(src).mean()
pred = np.array([1 if (maj[s] * (sum(src == s)) - y[i]) / (sum(src == s) - 1) >= .5 else 0 for i, s in enumerate(src)])
print('\nSource-only majority predictor (leave-one-flake-out) acc: %.3f' % (pred == y).mean())
tab = pd.crosstab(src, t.meta.label); tab['frac_PP'] = (tab['PP'] / (tab['PE'] + tab['PP'])).round(2)
print(tab)
tab.to_csv(RES / 'tables' / 'exp4_source_class_balance.csv')
json_out = {'source_only_majority_acc': float((pred == y).mean())}
save_json(json_out, RES / 'tables' / 'exp4_confound.json')
