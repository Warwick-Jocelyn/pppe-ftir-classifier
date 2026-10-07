"""Exp 6 - efficiency: (a) how little spectrum is needed, (b) how little training data is needed,
(c) an interpretable 2-band-ratio rule, (d) wavenumber attribution of the linear model."""
import sys, warnings, time; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from evalkit import *
from models import *
from preprocess import SpectraPreproc, REGIONS, NATIVE_WN

t = get_task(); y = t.y
key = pd.factorize(strat_key(t.meta))[0]
splits = repeated_cv_splits(y, n_splits=5, n_repeats=5, strat=key)
loso = loso_splits(t)

# ---------------------------------------------------------------- (a) minimal spectral windows / resolution
WIN = dict(REGIONS)   # explicit lists are passed to workers (module-level dict edits would not reach loky processes)
WIN.update({
    'pe_doublet_only': [(700, 740)],
    'pp_1376_only': [(1340, 1400)],
    'two_windows': [(700, 740), (1340, 1400)],
    'ch_stretch_only': [(2800, 3000)],
})
rows = []
for reg in ['full', 'fp+ch', 'fingerprint', 'ch_stretch_only', 'narrow', 'two_windows', 'pe_doublet_only', 'pp_1376_only']:
    for step in [1, 2, 4, 8, 16]:
        pp = dict(region=WIN[reg], step=step, smooth=11 if step <= 2 else 0, deriv=1 if step <= 2 else 0, norm='snv')
        for mn in ['logreg', 'plsda']:
            est = make_classical(mn, pp)
            n_feat = len(SpectraPreproc(**pp).get_wn_out())
            r = {}
            for pname, spl in [('rcv', splits), ('loso', loso)]:
                P = oof_predict(est, t.X, y, spl)
                s = summarise_oof(P, y, t.black); r[pname] = s
            rows.append(dict(region=reg, step=step, model=mn, n_points=n_feat, eff_res_cm=step * 0.964,
                             rcv_acc=r['rcv']['acc'], rcv_errors=r['rcv']['errors'], rcv_auc=r['rcv']['auc'],
                             loso_acc=r['loso']['acc'], loso_auc=r['loso']['auc'], black_acc=r['rcv'].get('black_acc', np.nan)))
    print(f'window {reg} done', flush=True)
pd.DataFrame(rows).to_csv(RES / 'tables' / 'exp6a_min_spectrum.csv', index=False)

# ---------------------------------------------------------------- (b) learning curve
dev, test = holdout_split(t)
sizes = [8, 12, 16, 24, 40, 64, len(dev)]
def lc_job(name, est, n, seed):
    rng = np.random.default_rng(seed)
    # stratified subsample of dev (at least 2 per class)
    idx = np.concatenate([rng.choice(dev[y[dev] == c], max(4, int(round(n * (y[dev] == c).mean()))), replace=False) for c in (0, 1)]) if n < len(dev) else dev
    m = clone(est).fit(t.X[idx], y[idx])
    p = m.predict_proba(t.X[test])[:, 1]
    return dict(model=name, n_train=len(idx), seed=seed, acc=float(((p >= .5) == y[test]).mean()), auc=metrics_from_proba(y[test], p)['auc'])
lc_models = {'logreg': make_classical('logreg'), 'plsda': make_classical('plsda'), 'rbfsvm': make_classical('rbfsvm'),
             'extratrees': make_classical('extratrees'), 'colour-only rf': make_colour('rf')}
jobs = [(nm, e, n, s) for nm, e in lc_models.items() for n in sizes for s in range(20)]
res = Parallel(n_jobs=-1)(delayed(lc_job)(*j) for j in jobs)
pd.DataFrame(res).to_csv(RES / 'tables' / 'exp6b_learning_curve.csv', index=False)
print('learning curve done', flush=True)

# ---------------------------------------------------------------- (c) two-band-ratio rule (interpretable)
X = t.X[:, :N_SPEC]
def band(c, hw=6): return X[:, (NATIVE_WN > c - hw) & (NATIVE_WN < c + hw)].max(1)
F = np.column_stack([band(720) / band(1463), band(1376) / band(1463)])
F2 = np.column_stack([F, band(2951) / band(2918)])
out = {}
for nm, FF in [('2 ratios (720/1463, 1376/1463)', F), ('3 ratios (+2951/2918)', F2)]:
    pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=10))])
    for pname, spl in [('rcv', splits), ('loso', loso)]:
        P = oof_predict(pipe, FF, y, spl)
        s = summarise_oof(P, y, t.black); out[f'{nm} | {pname}'] = {k: s[k] for k in ['acc', 'acc_sd', 'auc', 'errors']}
save_json(out, RES / 'tables' / 'exp6c_band_ratio_rule.json')
print(pd.DataFrame(out).T.round(4))

# ---------------------------------------------------------------- (d) linear-model attribution
est = make_classical('logreg', dict(region='full', step=1, smooth=11, deriv=0, norm='snv')).fit(t.X, y)
pp = est.named_steps['pp']; clf = est.named_steps['clf'].best_estimator_; sc = est.named_steps['sc']
coef = clf.coef_.ravel() / sc.scale_
np.save(RES / 'cache' / 'logreg_coef_full.npy', np.vstack([pp.get_wn_out(), coef]))
print('attribution saved; best C =', est.named_steps['clf'].best_params_)
