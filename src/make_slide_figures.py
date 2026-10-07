"""Figures made specifically for the results slide deck (clean, large-font, consistent palette)."""
import sys, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from evalkit import *

OUT = RES / 'figures' / 'slides'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
PE, PP, INK, TRAIN, TEST, GREY = '#1F6FB5', '#D9620B', '#1B2430', '#C9D3E3', '#2A9D8F', '#8A94A6'

# ------------------------------------------------------------------ split figure
t = get_task(); m = t.meta.copy()
order = ['Impact_Float_PP', 'Impact_Float_PE', 'Impact_Sink_PP', 'Impact_Sink_PE', 'Impact_Mixed_PO',
         'Circular 11_Aylesbury_PE_Pallet', 'Circular 11_Express_Bag_13_PE/PP_Pallet', 'Circular 11_Express_Bag_17_PE/PP_Pallet', 'Circular 11_Express_Bag_18_PE/PP_Pallet']
short = ['Impact · Float PP', 'Impact · Float PE', 'Impact · Sink PP', 'Impact · Sink PE', 'Impact · Mixed PO',
         'Circ. 11 · Aylesbury PE', 'Circ. 11 · Express bag 13', 'Circ. 11 · Express bag 17', 'Circ. 11 · Express bag 18']
m['src_i'] = m.source.map({s: i for i, s in enumerate(order)})
m = m.sort_values(['src_i', 'label', 'sample_id']).reset_index(drop=True)
split = pd.read_csv(RES / 'splits' / 'holdout_split.csv').set_index('sample_id').split
fig = plt.figure(figsize=(12.3, 4.9))
axb = fig.add_axes([0.155, 0.16, 0.215, 0.70])
cnt = pd.crosstab(m.src_i, m.label).reindex(range(9)).fillna(0)
blk = m.groupby('src_i').is_black.sum().reindex(range(9)).fillna(0)
y = np.arange(9)[::-1]
axb.barh(y, cnt['PE'], color=PE, label='PE'); axb.barh(y, cnt['PP'], left=cnt['PE'], color=PP, label='PP')
for yi, (a, b, k) in zip(y, zip(cnt['PE'], cnt['PP'], blk)):
    axb.text(a + b + .6, yi, f'{int(a+b)}' + (f'  ({int(k)} black)' if k else ''), va='center', fontsize=10, color=INK)
axb.set_yticks(y); axb.set_yticklabels(short, fontsize=10.5); axb.set_xlim(0, 44); axb.set_xlabel('flakes'); axb.legend(frameon=False, loc='upper right', fontsize=10, bbox_to_anchor=(1.0, 1.08), ncol=2)
axb.axhline(3.5, color=GREY, lw=.8, ls=':')
fig.text(0.012, 0.93, 'A  178 flakes (98 PE / 80 PP) from 9 sources', fontsize=13, weight='bold', color=INK)

NC, NR = 22, 9
def panel(rect, title, sub, roles):
    ax = fig.add_axes(rect); ax.set_xlim(0, NC); ax.set_ylim(NR, 0); ax.set_aspect('equal'); ax.axis('off')
    for i, r in enumerate(roles):
        ax.add_patch(Rectangle((i % NC + .08, i // NC + .08), .84, .84, color=TEST if r == 1 else TRAIN, lw=0))
    ax.set_title(title, fontsize=11.5, weight='bold', color=INK, loc='left', pad=6)
    ax.text(0, NR + .7, sub, fontsize=9.5, color='#4A5568', va='top')
n = len(m); rng = np.random.default_rng(3)
def pad(r): return list(r) + [-1] * (NC * NR - len(r))
# 1 repeated CV: one illustrative fold (stratified random 20 %)
fold = np.zeros(n, int); fold[rng.choice(n, int(round(n * .2)), replace=False)] = 1
panel([0.415, 0.585, 0.255, 0.262], 'B  Repeated 5-fold CV  (×5)', 'each flake tested once, by a model that never saw it\n(5 folds, ~142 train / ~36 test; redone 5x; one fold shown)', pad(fold))
panel([0.715, 0.585, 0.255, 0.262], 'C  Fixed hold-out  (133 / 45)', 'one fixed split: train on 133, test on 45 unseen flakes\n(stratified by class × family × black)', pad((m.sample_id.map(split) == 'test').astype(int).values))
panel([0.415, 0.170, 0.255, 0.262], 'D  Leave-one-source-out  (9 folds)', 'train on 8 sources, test on the 9th, rotate through all 9\n(example: Express bag 13 held out)', pad((m.src_i == 6).astype(int).values))
panel([0.715, 0.170, 0.255, 0.262], 'E  Shift tests', 'train on one family, test on the other\n(Impact <-> Circular 11; non-black -> black)', pad((m.family == 'Circular11').astype(int).values))
fig.patches.append(Rectangle((0.415, 0.012), .012, .028, transform=fig.transFigure, color=TRAIN)); fig.text(0.432, 0.016, 'train', fontsize=10.5)
fig.patches.append(Rectangle((0.485, 0.012), .012, .028, transform=fig.transFigure, color=TEST)); fig.text(0.502, 0.016, 'test', fontsize=10.5)
fig.text(0.56, 0.016, 'each square = one flake, ordered by source as in A', fontsize=10.5, color='#4A5568')
fig.savefig(OUT / 'fig_split.png', dpi=200, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ robustness figure
a = pd.concat([pd.read_csv(RES / 'tables' / 'exp8_candidates_robustness.csv'), pd.read_csv(RES / 'tables' / 'exp8b_final_robustness.csv')])
lines = [('logreg', 'Logistic regression (no augmentation)', '#C9A227', '--', 2), ('rbfsvm', 'RBF-SVM (no augmentation)', GREY, '--', 2),
         ('AugSVM + colour (late fusion)', 'Deployed model + colour', TEST, '-.', 2.2), ('AugSVM (deployed, FTIR only)', 'Augmented RBF-SVM (deployed)', '#16324F', '-', 3.6)]
panels = [('noise', 'Detector noise (σ as fraction of peak)'), ('blur', 'Optical resolution loss (FWHM, cm$^{-1}$)'), ('shift', 'Wavenumber calibration shift (cm$^{-1}$)')]
fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), sharey=True)
for ax, (k, ttl) in zip(axes, panels):
    for key, lab, c, ls, lw in lines:
        g = a[(a.model == key) & (a.kind == k)].sort_values('level')
        ax.plot(range(len(g)), g.acc * 100, color=c, ls=ls, lw=lw, marker='o', ms=4, label=lab, zorder=5 if 'deployed' in key else 3)
    lv = sorted(a[a.kind == k].level.unique()); ax.set_xticks(range(len(lv))); ax.set_xticklabels([f'{v:g}' for v in lv]); ax.set_xlabel(ttl); ax.set_ylim(0, 103); ax.grid(alpha=.25)
axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(loc='lower left', fontsize=9.5, frameon=True, framealpha=.95)
axes[2].annotate('RBF-SVM collapses\n(10.7 %)', xy=(4, 10.7), xytext=(1.6, 22), fontsize=10, arrowprops=dict(arrowstyle='->', color=GREY), color='#4A5568')
axes[0].annotate('logistic regression\n58 %', xy=(6, 58.4), xytext=(3.2, 38), fontsize=10, arrowprops=dict(arrowstyle='->', color='#C9A227'), color='#7A6414')
fig.tight_layout(); fig.savefig(OUT / 'fig_robustness.png', dpi=200, facecolor='white'); plt.close(fig)
print('ok', list(OUT.glob('*.png')))

# ------------------------------------------------------------------ title-slide graphic: real mean spectra on dark background
from preprocess import NATIVE_WN
X = t.X[:, :len(NATIVE_WN)]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
fig.patch.set_alpha(0); ax.patch.set_alpha(0)
w = (NATIVE_WN >= 650) & (NATIVE_WN <= 3100)
for lab, c in [('PE', '#5AA7EE'), ('PP', '#FF9A52')]:
    mu = X[t.meta.label.values == lab].mean(0)[w]
    ax.fill_between(NATIVE_WN[w], mu, color=c, alpha=.22, lw=0); ax.plot(NATIVE_WN[w], mu, color=c, lw=1.8, label=lab)
ax.set_xlim(3100, 650); ax.set_ylim(0, .46); ax.axis('off')
ax.text(2915, .43, 'C–H stretch', color='#CBD5E1', fontsize=10, ha='center'); ax.text(1376, .20, 'PP 1376', color='#FF9A52', fontsize=10, ha='center')
ax.text(720, .19, 'PE 720/730', color='#5AA7EE', fontsize=10, ha='center')
fig.savefig(OUT / 'fig_title_spectra.png', dpi=220, transparent=True); plt.close(fig)
print('title fig ok')

# ------------------------------------------------------------------ real-time budget figure (log scale)
fig, ax = plt.subplots(figsize=(8.6, 4.3))
items = [('Model: raw 6,845-pt spectrum → P(PP)\n(crop, derivative, SNV, SVM)', 1.33, '#16324F'),
         ('Read native OPUS .0 file', 1.4, TEST),
         ('Read text .dpt file', 11.5, TEST),
         ('Whole web request, 1 spectrum\n(browser round trip)', 10.0, GREY)]
for i, (lab, v, c) in enumerate(items):
    ax.barh(i, v, color=c, height=.55); ax.text(v * 1.15, i, f'{v:g} ms', va='center', fontsize=11, color=INK)
ax.barh(len(items), 10000 - 100, left=100, color='#E5E9F0', hatch='///', edgecolor='#B5BECB', height=.55)
ax.text(125, len(items), 'typical FTIR measurement\n~0.1 s to ~10 s', va='center', ha='left', fontsize=9.5, color='#334155', bbox=dict(facecolor='#E5E9F0', edgecolor='none', pad=1.5), zorder=6)
ax.axvline(1000, color=PP, ls='--', lw=1.3, zorder=1); ax.text(1100, -0.62, '1 s', color=PP, fontsize=10)
ax.set_yticks(range(len(items) + 1)); ax.set_yticklabels([x[0] for x in items] + ['Spectrum acquisition\n(reference range)'], fontsize=10.5)
ax.set_xscale('log'); ax.set_xlim(0.5, 20000); ax.set_xlabel('time per spectrum (ms, log scale)'); ax.set_ylim(-.8, len(items) + .5)
fig.tight_layout(); fig.savefig(OUT / 'fig_realtime.png', dpi=200, facecolor='white'); plt.close(fig)
print('realtime fig ok')
