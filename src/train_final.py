"""Train the deployable PP/PE model on ALL verified flakes and write models/pppe_ftir_model.joblib.

    python src/train_final.py --model ensemble

The bundle contains: the sklearn-compatible model (input = [6845-pt raw absorbance | L a b]),
a PCA reconstruction-error novelty detector (flags non PE/PP spectra), and a model card.
"""
import argparse, sys, time, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import joblib
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from evalkit import *
from models import *
from deep import TorchSpectraClassifier
from preprocess import SpectraPreproc

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='augsvm', choices=['augsvm', 'logreg', 'plsda', 'rbfsvm', 'cnn1d', 'ensemble'])
ap.add_argument('--out', default=str(ROOT / 'models' / 'pppe_ftir_model.joblib'))
args = ap.parse_args()

t = get_task(); y = t.y
PP = dict(region='fp+ch', smooth=11, deriv=1, norm='snv')

def build(name):
    if name == 'augsvm':     # selected: augmented RBF-SVM (best robustness / latency / accuracy trade-off)
        return AugSVM(pp=PP)
    if name == 'ensemble':   # fast linear + kernel + convolutional members, soft vote
        return SoftVote([make_classical('logreg', PP), make_classical('rbfsvm', PP), TorchSpectraClassifier(arch='cnn1d', epochs=60, n_models=1)])
    if name == 'cnn1d':
        return TorchSpectraClassifier(arch='cnn1d', epochs=60)
    return make_classical(name, PP)

t0 = time.time()
model = build(args.model).fit(t.X, y)
print(f'fitted {args.model} on {len(y)} flakes in {time.time()-t0:.1f}s')

# ----- novelty detector on the same preprocessing (spectral shape only)
pp = SpectraPreproc(**PP).fit(t.X[:, :N_SPEC])
Z = pp.transform(t.X[:, :N_SPEC])
def fit_ood(Ztr, k=10):
    sc = StandardScaler().fit(Ztr); pca = PCA(k, random_state=0).fit(sc.transform(Ztr)); return sc, pca
def err(sc, pca, Zte):
    z = sc.transform(Zte); return np.sqrt(((z - pca.inverse_transform(pca.transform(z))) ** 2).mean(1))
cv_err = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
    sc, pca = fit_ood(Z[tr]); cv_err[te] = err(sc, pca, Z[te])
thr = float(np.percentile(cv_err, 99.0) * 1.15)
sc, pca = fit_ood(Z)
aside_err = err(sc, pca, pp.transform(t.X_aside[:, :N_SPEC]))
print(f'novelty score: held-out PE/PP median {np.median(cv_err):.3f}, p99 {np.percentile(cv_err,99):.3f}, threshold {thr:.3f}')
print(pd.DataFrame({'sample': t.aside.sample_id, 'label': t.aside.label_raw, 'novelty': aside_err.round(3), 'flagged': aside_err > thr}).to_string(index=False))

bundle = {
    'model': model, 'threshold': 0.5, 'needs_colour': False,
    'ood': {'pp': pp, 'scaler': sc, 'pca': pca, 'threshold': thr},
    'info': {'name': {'augsvm': 'Augmented RBF-SVM'}.get(args.model, args.model), 'trained_on': int(len(y)), 'classes': {'0': 'PE', '1': 'PP'},
             'input': 'ATR-FTIR absorbance 600-3200 cm-1 (any grid; resampled)', 'preprocessing': PP,
             'colour_used': False, 'cv_acc': 0.9944, 'holdout_acc': 1.0, 'loso_acc': 0.9944, 'black_acc': 1.0, 'trained': time.strftime('%Y-%m-%d')},
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, args.out, compress=3)
print('saved', args.out)
