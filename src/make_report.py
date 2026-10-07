"""Build all result figures + tables from the experiment outputs in results/."""
import sys, pickle, json, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from evalkit import *
from preprocess import NATIVE_WN

FIG, TAB = RES / 'figures', RES / 'tables'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
C = {'PE': '#1f77b4', 'PP': '#d95f02', 'FTIR': '#2b59c3', 'Colour': '#8c564b', 'Fusion': '#2ca02c', 'DL': '#9467bd'}
def fam(name):
    return 'Colour' if name.startswith('Colour') else 'Fusion' if 'usion' in name else 'DL' if name.startswith('DL') else 'FTIR'

t = get_task(); y = t.y

# ------------------------------------------------------------------ 06 benchmark overview
frames = [pd.read_csv(TAB / 'exp2_benchmark_full.csv')]
if (TAB / 'exp3_deep_full.csv').exists(): frames.append(pd.read_csv(TAB / 'exp3_deep_full.csv'))
bm = pd.concat(frames, ignore_index=True)
bm.to_csv(TAB / 'benchmark_all_models.csv', index=False)
wide = bm.pivot(index='model', columns='protocol', values='acc')
wide['black_rcv'] = bm[bm.protocol == 'rcv5x5'].set_index('model')['black_acc']
wide['auc_rcv'] = bm[bm.protocol == 'rcv5x5'].set_index('model')['auc']
wide['errors_rcv'] = bm[bm.protocol == 'rcv5x5'].set_index('model')['errors']
wide = wide.sort_values(['rcv5x5', 'loso', 'holdout'], ascending=False)
wide.round(4).to_csv(TAB / 'benchmark_summary_wide.csv')
order = wide.index.tolist()[::-1]
fig, ax = plt.subplots(figsize=(11, 0.34 * len(order) + 2))
ypos = np.arange(len(order)); h = 0.26
for k, (pr, lab) in enumerate([('rcv5x5', 'Repeated 5-fold CV'), ('loso', 'Leave-one-source-out'), ('holdout', 'Hold-out 25%')]):
    ax.barh(ypos + (k - 1) * h, [wide.loc[m, pr] for m in order], h, label=lab, color=['#4c72b0', '#dd8452', '#55a868'][k])
ax.set_yticks(ypos); ax.set_yticklabels(order); ax.set_xlim(0.4, 1.005); ax.axvline(0.5, c='k', lw=.5)
for lab in ax.get_yticklabels(): lab.set_color(C[fam(lab.get_text())])
ax.set_xlabel('Accuracy'); ax.legend(loc='lower left'); ax.set_title('PP vs PE: accuracy of every model under three protocols (colour of label = input type)')
fig.tight_layout(); fig.savefig(FIG / '06_benchmark_accuracy.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 07 ROC + confusion (hold-out + rcv pooled)
oof = pickle.load(open(RES / 'cache' / 'exp2_oof.pkl', 'rb'))
dev, test = oof['dev'], oof['test']
sel = ['FTIR|logreg', 'FTIR|plsda', 'FTIR|rbfsvm', 'FTIR|extratrees', 'Colour|rf', 'Late-fusion|logreg+Lab']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for m in sel:
    P = oof['oof'][(m, 'rcv5x5')]; p = np.nanmean(P, 0)
    fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(y, p):.3f})", lw=1.6)
ax[0].plot([0, 1], [0, 1], 'k:', lw=.8); ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate'); ax[0].set_title('ROC - repeated CV (out-of-fold)'); ax[0].legend(fontsize=7, loc='lower right')
for a, (m, ttl) in zip(ax[1:], [('FTIR|logreg', 'FTIR logistic regression'), ('Colour|rf', 'Colour-only random forest')]):
    p = np.nanmean(oof['oof'][(m, 'rcv5x5')], 0)
    ConfusionMatrixDisplay(confusion_matrix(y, p >= .5), display_labels=['PE', 'PP']).plot(ax=a, colorbar=False, cmap='Blues'); a.set_title(f'{ttl}\nCV out-of-fold (n={len(y)})')
fig.tight_layout(); fig.savefig(FIG / '07_roc_confusion.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 08 shift / black
f = TAB / 'exp4_shift_black.csv'
if f.exists():
    d = pd.read_csv(f)
    d['grp'] = d.exp.str.split(':').str[0]
    exps = d.exp.unique()
    fig, axes = plt.subplots(1, len(exps), figsize=(4.8 * len(exps), 4.8), sharey=True)
    for a, e in zip(np.atleast_1d(axes), exps):
        s = d[d.exp == e].set_index('model')['acc']
        a.barh(range(len(s)), s.values, color=[C[fam(n.replace('FTIR ', 'FTIR|').replace('Colour ', 'Colour|'))] if n.startswith(('FTIR', 'Colour')) else C['Fusion'] for n in s.index])
        a.set_yticks(range(len(s))); a.set_yticklabels(s.index); a.invert_yaxis(); a.set_xlim(0, 1.02); a.set_title(e.replace(': ', ':\n'), fontsize=9); a.set_xlabel('Accuracy')
        for i, v in enumerate(s.values): a.text(min(v + .01, .93), i, f'{v:.2f}', va='center', fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / '08_shift_and_black.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 09 robustness
f = TAB / 'exp5_robustness.csv'
if f.exists():
    r = pd.read_csv(f)
    kinds = [('noise', 'White noise (sigma / peak)'), ('baseline', 'Baseline tilt+offset (x peak)'), ('attenuation', 'Signal gain (instrument noise fixed)'),
             ('blur', 'Optical resolution loss, FWHM (cm$^{-1}$)'), ('shift', 'Wavenumber calibration shift (cm$^{-1}$)')]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4), sharey=True)
    for a, (k, ttl) in zip(axes, kinds):
        for mname, g in r[r.kind == k].groupby('model'):
            g = g.sort_values('level', ascending=(k != 'attenuation'))
            a.plot(range(len(g)), g.acc, marker='o', ms=3.5, lw=1.5, label=mname,
                   ls='--' if ('aug' in mname or 'fusion' in mname or 'colour' in mname) else '-')
        lv = sorted(r[r.kind == k].level.unique(), reverse=(k == 'attenuation'))
        a.set_xticks(range(len(lv))); a.set_xticklabels([f'{v:g}' for v in lv], fontsize=8); a.set_title(ttl, fontsize=10); a.set_ylim(0.4, 1.01); a.grid(alpha=.3)
    axes[0].set_ylabel('Accuracy (CV, clean-trained)'); axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(FIG / '09_robustness.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 10 minimal spectrum
f = TAB / 'exp6a_min_spectrum.csv'
if f.exists():
    m = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for a, (mod, ttl) in zip(ax, [('logreg', 'Logistic regression'), ('plsda', 'PLS-DA')]):
        pv = m[m.model == mod].pivot(index='region', columns='step', values='loso_acc')
        pv = pv.loc[['full', 'fp+ch', 'fingerprint', 'ch_stretch_only', 'narrow', 'two_windows', 'pe_doublet_only', 'pp_1376_only']]
        im = a.imshow(pv.values, vmin=.5, vmax=1, cmap='viridis', aspect='auto')
        a.set_xticks(range(pv.shape[1])); a.set_xticklabels([f'every {s}\n(~{s*0.964:.0f} cm$^{{-1}}$)' for s in pv.columns], fontsize=8)
        a.set_yticks(range(pv.shape[0])); a.set_yticklabels(pv.index)
        npts = m[m.model == mod].pivot(index='region', columns='step', values='n_points').loc[pv.index]
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]): a.text(j, i, f'{pv.values[i,j]:.3f}\n{int(npts.values[i,j])} pts', ha='center', va='center', color='w' if pv.values[i, j] < .85 else 'k', fontsize=7)
        a.set_title(f'{ttl}: leave-one-source-out accuracy')
    fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(FIG / '10_min_spectrum.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 11 learning curve
f = TAB / 'exp6b_learning_curve.csv'
if f.exists():
    lc = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for mname, g in lc.groupby('model'):
        s = g.groupby('n_train')[['acc', 'auc']].agg(['mean', 'std'])
        for a, k in zip(ax, ['acc', 'auc']):
            a.errorbar(s.index, s[(k, 'mean')], yerr=s[(k, 'std')], marker='o', ms=3.5, capsize=2, label=mname, lw=1.4)
    for a, k in zip(ax, ['Accuracy', 'ROC-AUC']): a.set_xscale('log'); a.set_xlabel('Training flakes'); a.set_ylabel(k + ' on fixed hold-out'); a.grid(alpha=.3)
    ax[0].legend(fontsize=8); fig.suptitle('Learning curve (20 random subsets per size)'); fig.tight_layout(); fig.savefig(FIG / '11_learning_curve.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 12 attribution
f = RES / 'cache' / 'logreg_coef_full.npy'
if f.exists():
    wn, coef = np.load(f)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=False)
    ax[0].plot(wn, coef, c='k', lw=.8); ax[0].axhline(0, c='grey', lw=.5); ax[0].set_xlim(3100, 600); ax[0].set_ylabel('logistic coef (PP+ / PE-)')
    ax[0].set_title('Which wavenumbers drive the PP-vs-PE decision? (logistic regression on SNV spectra)')
    X = t.X[:, :N_SPEC]
    for lab in ['PE', 'PP']: ax[1].plot(NATIVE_WN, X[t.meta.label.values == lab].mean(0), c=C[lab], label=f'mean {lab}')
    ax[1].set_xlim(3100, 600); ax[1].set_xlabel('Wavenumber (cm$^{-1}$)'); ax[1].set_ylabel('Absorbance'); ax[1].legend()
    for a in ax:
        for x0, x1, n in [(700, 740, 'PE 720/730'), (1360, 1392, 'PP 1376'), (2940, 2962, 'PP 2951'), (960, 1000, 'PP 973/998'), (1150, 1175, 'PP 1166')]:
            a.axvspan(x0, x1, color='gold', alpha=.25);
    ax[0].set_xlim(3100, 600); ax[1].set_xlim(3100, 600)
    fig.tight_layout(); fig.savefig(FIG / '12_attribution.png', dpi=130); plt.close(fig)
print('figures written:', sorted(p.name for p in FIG.glob('*.png')))

# ------------------------------------------------------------------ 14 deployment candidates under stress
fa, fb = TAB / 'exp8_candidates_robustness.csv', TAB / 'exp8b_final_robustness.csv'
if fa.exists() and fb.exists():
    r = pd.concat([pd.read_csv(fa), pd.read_csv(fb)], ignore_index=True)
    keep = {'logreg': 'logreg (no aug)', 'rbfsvm': 'RBF-SVM (no aug)', 'AUG logreg': 'logreg + aug', 'AUG plsda': 'PLS-DA + aug', 'AUG extratrees': 'ExtraTrees + aug',
            'ENS logreg+rbfsvm+cnn1d': 'ensemble (no aug)', 'AugSVM (deployed, FTIR only)': 'AugSVM (DEPLOYED)', 'AugSVM + colour (late fusion)': 'AugSVM + colour',
            'colour-only rf': 'colour only'}
    r = r[r.model.isin(keep)].copy(); r['model'] = r.model.map(keep)
    sty = {'AugSVM (DEPLOYED)': dict(c='#d62728', lw=3, zorder=5), 'AugSVM + colour': dict(c='#2ca02c', lw=2, ls='--'), 'colour only': dict(c='#8c564b', lw=1.5, ls=':'),
           'RBF-SVM (no aug)': dict(c='#7f7f7f', lw=1.5, ls='--'), 'logreg (no aug)': dict(c='#bcbd22', lw=1.5, ls='--')}
    kinds = [('noise', 'White noise (sigma / peak)'), ('attenuation', 'Signal gain'), ('blur', 'Resolution loss, FWHM (cm$^{-1}$)'), ('shift', 'Wavenumber shift (cm$^{-1}$)')]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    for a, (k, ttl) in zip(axes, kinds):
        for mname, g in r[r.kind == k].groupby('model'):
            g = g.sort_values('level', ascending=(k != 'attenuation')); a.plot(range(len(g)), g.acc, marker='o', ms=3.5, label=mname, **sty.get(mname, dict(lw=1.2, alpha=.8)))
        lv = sorted(r[r.kind == k].level.unique(), reverse=(k == 'attenuation')); a.set_xticks(range(len(lv))); a.set_xticklabels([f'{v:g}' for v in lv]); a.set_title(ttl); a.set_ylim(.4, 1.01); a.grid(alpha=.3)
    axes[0].set_ylabel('Accuracy (CV, trained on clean spectra)'); axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle('Deployment candidates under acquisition stress: training-time augmentation fixes the weak spots; adding colour does not help'); fig.tight_layout()
    fig.savefig(FIG / '14_deployment_candidates_robustness.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 15 latency
f = TAB / 'exp9_latency.csv'
if f.exists():
    L = pd.read_csv(f).sort_values('lat_median_ms')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].barh(L.model, L.lat_median_ms, xerr=[np.zeros(len(L)), (L.lat_p95_ms - L.lat_median_ms).clip(lower=0)], color=['#d62728' if 'deployed' in m else '#4c72b0' for m in L.model])
    ax[0].set_xlabel('ms per spectrum (median, bar to p95; 1 CPU thread; raw 6845-pt vector -> P(PP))'); ax[0].invert_yaxis(); ax[0].set_title('Single-spectrum inference latency')
    ax[1].barh(L.model, L.batch100_spectra_per_s, color=['#d62728' if 'deployed' in m else '#4c72b0' for m in L.model]); ax[1].set_xscale('log'); ax[1].invert_yaxis(); ax[1].set_xlabel('spectra / s (batch of 100)'); ax[1].set_yticklabels([])
    ax[1].set_title('Batch throughput')
    fig.tight_layout(); fig.savefig(FIG / '15_latency.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 16 final model: probabilities per flake
f = TAB / 'final_model_oof_predictions.csv'
if f.exists():
    p = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    rng = np.random.default_rng(0)
    for lab, c in [('PE', C['PE']), ('PP', C['PP'])]:
        s = p[p.label == lab]; ax[0].scatter(rng.uniform(-.25, .25, len(s)) + (0 if lab == 'PE' else 1), s.p_pp_oof, c=c, s=np.where(s.is_black, 70, 22), edgecolor=np.where(s.is_black, 'k', 'none'), alpha=.8)
    ax[0].axhline(.5, c='k', ls=':'); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['verified PE', 'verified PP']); ax[0].set_ylabel('out-of-fold P(PP)'); ax[0].set_title('Deployed model, repeated-CV probabilities (large dots with outline = black flakes)')
    bad = p[~p.correct]
    for _, r_ in bad.iterrows(): ax[0].annotate(r_.sample_id, (1, r_.p_pp_oof), textcoords='offset points', xytext=(-110, 10), fontsize=8, arrowprops=dict(arrowstyle='->'))
    ps = pd.read_csv(TAB / 'final_model_per_source.csv'); ax[1].barh(ps.source.str.replace('Circular 11_', 'C11 ').str.replace('_PE/PP_Pallet', ''), ps.acc, color='#2b59c3'); ax[1].set_xlim(.8, 1.005); ax[1].invert_yaxis(); ax[1].set_xlabel('CV accuracy')
    for i, (a_, n_) in enumerate(zip(ps.acc, ps.n)): ax[1].text(.805, i, f'n={n_}', va='center', color='w', fontsize=8)
    ax[1].set_title('Accuracy by sample batch'); fig.tight_layout(); fig.savefig(FIG / '16_final_model_probabilities.png', dpi=130); plt.close(fig)
print('figures:', sorted(p.name for p in FIG.glob('*.png')))

# ------------------------------------------------------------------ 00 preprocessing ablation
f = TAB / 'exp1_preprocessing.csv'
if f.exists():
    e = pd.read_csv(f); pv = e.pivot_table(index='prep', columns='region', values='acc', aggfunc='mean')
    pv = pv[['full', 'fp+ch', 'fingerprint', 'ch', 'narrow']].loc[['raw(no norm)', 'max', 'SNV', 'poly+SNV', 'ALS+SNV', 'SG1(11)+SNV', 'SG2(15)+SNV']]
    fig, ax = plt.subplots(figsize=(8, 4.2)); im = ax.imshow(pv.values, vmin=.985, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:.4f}', ha='center', va='center', color='w' if (np.isnan(v) or v < .992) else 'k', fontsize=8)
    ax.set_title('Preprocessing ablation: CV accuracy (mean of logreg / PLS-DA / RBF-SVM)'); fig.colorbar(im); fig.tight_layout(); fig.savefig(FIG / '00_preprocessing_ablation.png', dpi=130); plt.close(fig)
