"""Exp 7 - how does the PREVIOUS model (Mix-plastic-composition-prediction exp12: PLSR predicting HDPE% in HDPE/LDPE
mixtures, 600-3000 cm-1, every 10th point -> 498 features) behave on this PE/PP flake dataset?

Caveat (documented in the report): the shipped bundle stores no wavenumber axis. Its 498 features = 600-3000 cm-1
subsampled; we therefore interpolate each spectrum onto 498 equispaced points in 600-3000 cm-1 (4.82 cm-1 step) -
the best reconstruction possible, but approximate. A bundle is only safe to unpickle because we inspected it
statically first (only sklearn / numpy / the repo's own classes)."""
import sys, warnings, types; sys.dont_write_bytecode = True; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
REPO = '/home/supervisor/Jocelyn/Polyflow/Mix-plastic-composition-prediction'
sys.path.insert(0, REPO)
import json, numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import ftir_ML                                   # defines ModelBundle (pickle refers to __main__.ModelBundle)
sys.modules['__main__'].ModelBundle = ftir_ML.ModelBundle
from preprocessing import transform_with_preprocessors
from evalkit import *
from preprocess import NATIVE_WN

t = get_task(); y = t.y
b = joblib.load(f'{REPO}/runs/exp12/bundle.joblib')
print('old bundle:', b.model_name, b.preprocess_names, 'features', b.feature_len, 'wn', b.wn_start, b.wn_end, 'sample', b.sample)
grid = np.linspace(600, 3000, b.feature_len)
X = np.vstack([np.interp(grid, NATIVE_WN, s) for s in t.X[:, :len(NATIVE_WN)]])
pred = np.ravel(b.model.predict(transform_with_preprocessors(X, b.preprocessors, b.feature_len)))
df = t.meta[['sample_id', 'label', 'source', 'colour_name']].assign(pred_hdpe_pct=pred)
df.to_csv(RES / 'tables' / 'exp7_old_model_predictions.csv', index=False)
auc = roc_auc_score(y == 1, pred)  # old-model output used as a PP score (higher = more PP-like)
fpr, tpr, thr = roc_curve(y == 1, pred); j = np.argmax(tpr - fpr)
summary = {'n': len(y), 'auc_PP_score': float(auc), 'pred_pct_PE_mean': float(pred[y == 0].mean()), 'pred_pct_PE_sd': float(pred[y == 0].std()),
           'pred_pct_PP_mean': float(pred[y == 1].mean()), 'pred_pct_PP_sd': float(pred[y == 1].std()),
           'acc_at_nominal_50pct_(HDPE>=50 -> PE)': float(((pred >= 50) == (y == 0)).mean()), 'youden_thr_(fit on same data, optimistic)': float(thr[j]), 'acc_at_youden_thr_(optimistic)': float(((pred >= thr[j]) == (y == 1)).mean()),
           'frac_outside_0_100': float(((pred < 0) | (pred > 100)).mean())}
save_json(summary, RES / 'tables' / 'exp7_old_model_summary.json'); print(json.dumps(summary, indent=1) if False else summary)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = np.linspace(min(pred.min(), -50), max(pred.max(), 150), 40)
ax[0].hist(pred[y == 0], bins, alpha=.7, label='PE (verified)', color='#1f77b4'); ax[0].hist(pred[y == 1], bins, alpha=.7, label='PP (verified)', color='#d95f02')
ax[0].axvspan(0, 100, color='grey', alpha=.1); ax[0].set_xlabel('Old model output: predicted "HDPE %"'); ax[0].set_ylabel('flakes'); ax[0].legend()
ax[0].set_title(f'Previous HDPE/LDPE regressor on PE/PP flakes (AUC {auc:.2f}, acc@nominal 50%={summary["acc_at_nominal_50pct_(HDPE>=50 -> PE)"]:.2f})')
ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], 'k:'); ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR'); ax[1].set_title('ROC: old-model output as a PP score')
fig.tight_layout(); fig.savefig(RES / 'figures' / '13_old_model_on_new_data.png', dpi=130)
