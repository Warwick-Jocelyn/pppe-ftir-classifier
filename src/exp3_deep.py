"""Exp 3 - deep 1-D backbones under the same three protocols (+ augmentation ablation)."""
import sys, time, pickle, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from evalkit import *
from deep import TorchSpectraClassifier

t = get_task(); y = t.y
key = pd.factorize(strat_key(t.meta))[0]
dev, test = holdout_split(t)
protocols = {'rcv5x5': repeated_cv_splits(y, n_splits=5, n_repeats=3, strat=key),
             'holdout': [(0, dev, test)], 'loso': loso_splits(t)}
zoo = {f'DL|{a}': TorchSpectraClassifier(arch=a, epochs=80, augment=True) for a in ['mlp', 'cnn1d', 'resnet1d', 'transformer']}
zoo['DL|cnn1d (no aug)'] = TorchSpectraClassifier(arch='cnn1d', epochs=80, augment=False)
zoo['DL|resnet1d (no aug)'] = TorchSpectraClassifier(arch='resnet1d', epochs=80, augment=False)
rows, oof = [], {}
t0 = time.time()
for name, est in zoo.items():
    for pname, spl in protocols.items():
        P = oof_predict(est, t.X, y, spl, n_jobs=30)
        oof[(name, pname)] = P
        s = summarise_oof(P, y, t.black); s.update(model=name, protocol=pname); rows.append(s)
    r = [x for x in rows if x['model'] == name]
    print(f"{name:26s} " + "  ".join(f"{x['protocol']}:{x['acc']:.3f}" for x in r) + f"   ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(RES / 'tables' / 'exp3_deep_full.csv', index=False)
with open(RES / 'cache' / 'exp3_oof.pkl', 'wb') as f: pickle.dump({'oof': oof}, f)
