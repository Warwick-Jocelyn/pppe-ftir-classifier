"""Exp 9 - real-time suitability: single-spectrum end-to-end latency (raw native-grid vector -> P(PP)), throughput,
model size. Median / p95 over 200 calls on one CPU thread (the web service handles requests per worker)."""
import sys, time, warnings, os, tempfile; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, torch
torch.set_num_threads(1)
from evalkit import *
from models import *
from deep import TorchSpectraClassifier
from preprocess import SpectraPreproc, resample_to_native
from inference import parse_spectrum_bytes

t = get_task(); y = t.y
cands = {
 'logreg':        make_classical('logreg'),
 'plsda':         make_classical('plsda'),
 'rbfsvm':        make_classical('rbfsvm'),
 'extratrees':    make_classical('extratrees'),
 'lightgbm':      make_classical('lightgbm'),
 'cnn1d':         TorchSpectraClassifier(arch='cnn1d', epochs=60),
 'resnet1d':      TorchSpectraClassifier(arch='resnet1d', epochs=60),
 'transformer':   TorchSpectraClassifier(arch='transformer', epochs=60),
 'AugSVM (deployed)': AugSVM(),
 'AUG logreg':    AugmentedClassifier(make_classical('logreg')),
 'ENS AUGrbfsvm+AUGlogreg+cnn1d': SoftVote([AugmentedClassifier(make_classical('rbfsvm')), AugmentedClassifier(make_classical('logreg')), TorchSpectraClassifier(arch='cnn1d', epochs=60)]),
}
x1 = t.X[:1]; rows = []
for n, m in cands.items():
    t0 = time.perf_counter(); m.fit(t.X, y); fit_s = time.perf_counter() - t0
    for _ in range(10): m.predict_proba(x1)
    ts = []
    for i in range(200):
        xi = t.X[i % len(y):i % len(y) + 1]
        a = time.perf_counter(); m.predict_proba(xi); ts.append((time.perf_counter() - a) * 1e3)
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f: joblib.dump(m, f.name, compress=3); sz = os.path.getsize(f.name) / 1e6; os.unlink(f.name)
    xb = t.X[:100]; a = time.perf_counter(); m.predict_proba(xb); thr = 100 / (time.perf_counter() - a)
    rows.append(dict(model=n, fit_s=fit_s, lat_median_ms=np.median(ts), lat_p95_ms=np.percentile(ts, 95), batch100_spectra_per_s=thr, size_MB=sz))
    print(f'{n:34s} median {np.median(ts):7.2f} ms  p95 {np.percentile(ts,95):7.2f}  batch {thr:8.0f}/s  {sz:6.2f} MB', flush=True)
df = pd.DataFrame(rows); df.to_csv(RES / 'tables' / 'exp9_latency.csv', index=False)

# file parsing + resampling cost (what the web server also pays)
import glob
p = glob.glob(str(ROOT / 'Dataset/**/Float_PP_A1.0*'), recursive=True)
res = {}
for p_ in p:
    raw = open(p_, 'rb').read(); ts = []
    for _ in range(50):
        a = time.perf_counter(); w, ab, _ = parse_spectrum_bytes(raw); resample_to_native(w, ab); ts.append((time.perf_counter() - a) * 1e3)
    res[os.path.basename(p_)] = float(np.median(ts))
print('parse+resample median ms:', res)
save_json(res, RES / 'tables' / 'exp9_parse_ms.json')
