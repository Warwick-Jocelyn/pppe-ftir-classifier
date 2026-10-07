"""Exp 2 - model benchmark: FTIR-only classical models, colour-only, and FTIR+colour fusion,
under (a) repeated stratified 5-fold CV x5, (b) fixed 75/25 hold-out, (c) leave-one-source-out."""
import sys, time, pickle; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from evalkit import *
from models import *

t = get_task(); y = t.y
key = pd.factorize(strat_key(t.meta))[0]
dev, test = holdout_split(t)
protocols = {
  'rcv5x5':  repeated_cv_splits(y, n_splits=5, n_repeats=5, strat=key),
  'holdout': [(0, dev, test)],
  'loso':    loso_splits(t),
}
zoo = {}
for n in CLASSICAL: zoo[f'FTIR|{n}'] = make_classical(n)
for n in ['logreg', 'rbfsvm', 'rf']: zoo[f'Colour|{n}'] = make_colour(n)
for w in [1.0, 3.0, 10.0]: zoo[f'Early-fusion|logreg w={w:g}'] = EarlyFusion(colour_weight=w)
zoo['Late-fusion|logreg+Lab'] = LateFusion(ftir=make_classical('logreg'))
zoo['Late-fusion|svm+Lab'] = LateFusion(ftir=make_classical('rbfsvm'))

rows, oof = [], {}
t0 = time.time()
for name, est in zoo.items():
    for pname, spl in protocols.items():
        P = oof_predict(est, t.X, y, spl)
        oof[(name, pname)] = P
        s = summarise_oof(P, y, t.black); s.update(model=name, protocol=pname); rows.append(s)
    r = [x for x in rows if x['model'] == name]
    print(f"{name:34s} " + "  ".join(f"{x['protocol']}:{x['acc']:.3f}" for x in r) + f"   ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(RES / 'tables' / 'exp2_benchmark_full.csv', index=False)
with open(RES / 'cache' / 'exp2_oof.pkl', 'wb') as f: pickle.dump({'oof': oof, 'dev': dev, 'test': test}, f)
