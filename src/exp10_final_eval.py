"""Exp 10 - full evaluation of the selected deployment model (augmented RBF-SVM, FTIR only) + library baselines."""
import sys, warnings, pickle; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss
from evalkit import *
from models import *

t = get_task(); y = t.y
key = pd.factorize(strat_key(t.meta))[0]
dev, test = holdout_split(t)
MODEL = AugSVM()
protocols = {'rcv5x5': repeated_cv_splits(y, n_splits=5, n_repeats=5, strat=key), 'holdout': [(0, dev, test)], 'loso': loso_splits(t)}
fam = t.meta.family.values
protocols['train Impact -> test Circular11'] = [(0, np.where(fam == 'Impact')[0], np.where(fam == 'Circular11')[0])]
protocols['train Circular11 -> test Impact'] = [(0, np.where(fam == 'Circular11')[0], np.where(fam == 'Impact')[0])]
protocols['train non-black -> test black'] = [(0, np.where(~t.black)[0], np.where(t.black)[0])]
oof, rows = {}, []
for pn, spl in protocols.items():
    P = oof_predict(MODEL, t.X, y, spl); oof[pn] = P
    s = summarise_oof(P, y, t.black); m = ~np.isnan(P[0]); n = int(m.sum()); k = int(round(s['acc'] * n)) if P.shape[0] == 1 else None
    lo, hi = wilson_ci(int(((P[0][m] >= .5) != y[m]).sum() * -1 + n), n) if P.shape[0] == 1 else (np.nan, np.nan)
    rows.append(dict(protocol=pn, n_scored=n, acc=s['acc'], acc_sd_over_repeats=s['acc_sd'], acc_wilson95_lo=lo, acc_wilson95_hi=hi, bal_acc=s['bal_acc'], f1_pp=s['f1_pp'],
                     mcc=s['mcc'], auc=s['auc'], errors=s['errors'], black_acc=s.get('black_acc', np.nan), nonblack_acc=s.get('nonblack_acc', np.nan)))
    print(f"{pn:36s} acc {s['acc']:.4f}  errors {s['errors']:.2f}  auc {s['auc']:.4f}", flush=True)
res = pd.DataFrame(rows); res.to_csv(RES / 'tables' / 'exp10_final_model_eval.csv', index=False)

# per-flake OOF predictions (mean over CV repeats) + library baselines (NOT independent of the GT, see report)
pm = np.nanmean(oof['rcv5x5'], 0)
pred = t.meta[['sample_id', 'source', 'colour_name', 'is_black', 'label', 'pf_id', 'pf_score', 'os_id', 'os_score']].copy()
pred['p_pp_oof'] = pm; pred['pred'] = np.where(pm >= .5, 'PP', 'PE'); pred['correct'] = pred.pred == pred.label
pred.to_csv(RES / 'tables' / 'final_model_oof_predictions.csv', index=False)
print('misclassified (CV):'); print(pred[~pred.correct][['sample_id', 'label', 'colour_name', 'p_pp_oof', 'pf_score', 'os_score']].to_string(index=False))

base = []
for nm, col in [('Polymerflow library match', 'pf_id'), ('OpenSpecy library match', 'os_id')]:
    p = pred[col].str.upper().replace({'PE': 'PE', 'PP': 'PP'}); ok = p.isin(['PE', 'PP'])
    base.append(dict(system=nm, n=int(ok.sum()), acc=float((p[ok] == pred.label[ok]).mean()), errors=int((p[ok] != pred.label[ok]).sum()),
                     black_acc=float((p[ok & pred.is_black] == pred.label[ok & pred.is_black]).mean())))
base.append(dict(system='This model (FTIR, aug RBF-SVM) - CV', n=len(y), acc=float(pred.correct.mean()), errors=int((~pred.correct).sum()), black_acc=float(pred.correct[pred.is_black].mean())))
pd.DataFrame(base).to_csv(RES / 'tables' / 'baseline_comparison.csv', index=False); print(pd.DataFrame(base))

# per-source accuracy and calibration
ps = pred.groupby('source').agg(n=('correct', 'size'), acc=('correct', 'mean'), n_pp=('label', lambda s: (s == 'PP').sum()))
ps.to_csv(RES / 'tables' / 'final_model_per_source.csv'); print(ps)
save_json({'brier_cv': float(brier_score_loss(y, pm)), 'mean_conf_correct': float(np.maximum(pm, 1 - pm)[pred.correct].mean()),
           'min_conf_correct': float(np.maximum(pm, 1 - pm)[pred.correct].min())}, RES / 'tables' / 'final_model_calibration.json')
with open(RES / 'cache' / 'exp10_oof.pkl', 'wb') as f: pickle.dump({'oof': oof, 'dev': dev, 'test': test}, f)
