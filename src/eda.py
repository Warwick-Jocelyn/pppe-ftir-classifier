import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import build_dataset, ROOT
FIG=ROOT/'results'/'figures'; TAB=ROOT/'results'/'tables'
m,wn,X=build_dataset()
ok=m.label.isin(['PE','PP']).values
C={'PE':'#1f77b4','PP':'#d95f02'}
fig,ax=plt.subplots(2,2,figsize=(15,9))
for lab in ['PE','PP']:
    s=ok&(m.label==lab).values
    mu,sd=X[s].mean(0),X[s].std(0)
    for a,(lo,hi) in zip(ax.ravel(),[(400,7000),(600,1800),(2700,3100),(1300,1500)]):
        w=(wn>=lo)&(wn<=hi)
        a.plot(wn[w],mu[w],color=C[lab],label=f'{lab} (n={s.sum()})'); a.fill_between(wn[w],(mu-sd)[w],(mu+sd)[w],color=C[lab],alpha=.2)
        a.set_xlim(hi,lo); a.set_title(f'{lo}-{hi} cm$^{{-1}}$'); a.set_xlabel('Wavenumber (cm$^{-1}$)'); a.set_ylabel('Absorbance')
ax[0,0].legend(); fig.suptitle('Mean ± SD raw ATR-FTIR spectra by verified class'); fig.tight_layout(); fig.savefig(FIG/'01_mean_spectra_by_class.png',dpi=130); plt.close(fig)

# black vs non-black
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
w=(wn>=500)&(wn<=4000)
for a,(title,sel) in zip(ax,[('Black flakes',ok&m.is_black.values),('Non-black flakes (random 25/class)',None),('Max absorbance per spectrum',None)]):
    if sel is not None:
        for i in np.where(sel)[0]: a.plot(wn[w],X[i,w],color=C[m.label[i]],lw=.8,alpha=.7)
    elif title.startswith('Non'):
        rng=np.random.default_rng(0)
        for lab in ['PE','PP']:
            idx=np.where(ok&~m.is_black.values&(m.label==lab).values)[0]
            for i in rng.choice(idx,25,replace=False): a.plot(wn[w],X[i,w],color=C[lab],lw=.6,alpha=.6)
    if sel is not None or title.startswith('Non'):
        a.set_xlim(4000,500); a.set_title(title); a.set_xlabel('Wavenumber (cm$^{-1}$)'); a.set_ylabel('Absorbance'); a.set_ylim(-0.02,0.6)
mx=X[:,(wn>2800)&(wn<3000)].max(1)
d=pd.DataFrame({'max_CH':mx,'grp':np.where(m.is_black,'black','other'),'label':m.label})[ok]
for k,(g,sub) in enumerate(d.groupby(['grp','label'])): ax[2].scatter(np.full(len(sub),k)+np.random.default_rng(1).uniform(-.15,.15,len(sub)),sub.max_CH,s=14,c=C[g[1]])
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['black\nPE','black\nPP','other\nPE','other\nPP']); ax[2].set_ylabel('Peak absorbance 2800-3000 cm$^{-1}$'); ax[2].set_title('Signal strength (C-H stretch)')
fig.tight_layout(); fig.savefig(FIG/'02_black_vs_others.png',dpi=130); plt.close(fig)

# colour space
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
d=m[ok]
for lab in ['PE','PP']:
    s=d.label==lab
    ax[0].scatter(d.a[s],d.b[s],c=C[lab],s=16,alpha=.7,label=lab); ax[1].scatter(d.L[s],d.a[s],c=C[lab],s=16,alpha=.7); ax[2].scatter(d.L[s],d.b[s],c=C[lab],s=16,alpha=.7)
ax[0].set_xlabel('a*');ax[0].set_ylabel('b*');ax[1].set_xlabel('L*');ax[1].set_ylabel('a*');ax[2].set_xlabel('L*');ax[2].set_ylabel('b*'); ax[0].legend(); fig.suptitle('Colorimeter CIELAB by class'); fig.tight_layout(); fig.savefig(FIG/'03_colour_space.png',dpi=130); plt.close(fig)

# class balance by source
ct=pd.crosstab(m.source,m.label); ct.to_csv(TAB/'class_by_source.csv')
ax=ct.plot(kind='barh',stacked=True,figsize=(10,4.5),color=['#1f77b4','#bbb','#d95f02','#c0392b'][:ct.shape[1]]); plt.title('Verified label by source'); plt.tight_layout(); plt.savefig(FIG/'04_class_by_source.png',dpi=130); plt.close()
print(ct)
