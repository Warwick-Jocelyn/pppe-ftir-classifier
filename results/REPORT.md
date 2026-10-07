# PP vs PE classification from FTIR (+ colorimeter) — results report

Work package: real-time PP/PE flake identification from ATR-FTIR, with an assessment of whether colorimeter data helps on black polymers.
Everything here is reproducible from `src/` (see the project `README.md`). Figures are in `results/figures/`, all numbers in `results/tables/`.

## 1. Headline

| | |
|---|---|
| **Deployed model** | Augmented RBF-SVM on SG-1st-derivative + SNV spectra (600-1800 and 2700-3100 cm⁻¹), **FTIR only** |
| Repeated 5-fold CV (5×, n = 178) | **99.4 %** (177/178), balanced acc 0.994, MCC 0.989 |
| Fixed 25 % hold-out (n = 45) | **100 %** (45/45; Wilson 95 % CI 92.1-100 %) |
| Leave-one-source-out (9 sources) | **99.4 %** (177/178; CI 96.9-99.9 %) |
| Train Impact → test Circular 11 / reverse | 98.3 % (57/58) / 100 % (120/120) |
| **Black flakes (22: 17 PE, 5 PP)** | **22/22 in every protocol**, incl. trained on non-black flakes only |
| Latency | **1.3 ms** per spectrum (model, 1 CPU thread), ≈ 4 000 spectra/s batch; whole API call ≈ 10 ms round trip |
| Does colour help? | **No.** It adds nothing on top of FTIR, does not transfer between batches, and slightly hurts under noisy spectra |
| One persistent "error" | `C11_Exp_Bag_13_A4` (labelled PP): spectrally a **PE/PP blend** (Fig. 05). Every model, FTIR or not, gets it "wrong". |

**Read these caveats before quoting the numbers**
1. **The ground truth is not independent of the FTIR libraries.** The verified label (last column of the GT sheet) agrees with the Polymerflow library ID on every PE/PP flake (and with OpenSpecy, except the 3 "PE-dominant" ones), so "verified" ≈ "library match, checked". The libraries therefore score 100 % *by construction*; our model is learning to reproduce that expert/library judgement. It has not been validated against an independent method (e.g. DSC/TGA), which the sample-location sheet lists as planned measurements (HPS, DSC, TGA).
2. **Small, easy dataset.** 178 flakes, clean ATR spectra. Accuracy is saturated (all sensible models tie), so differences between backbones are *not* statistically meaningful; robustness, source-shift and efficiency are the discriminating tests.
3. **The 100 % hold-out is a validation split, not a pristine test.** Preprocessing choices were made by CV on all 178 flakes (the ablation shows they barely matter). A truly blind test needs a new batch.
4. Folder names are unreliable (`Float PE` is 16/24 PP). Only the verified label was used.

## 2. Data

* 187 labelled flakes (GT sheet last column "Varified Polymer ID"): PE 98, PP 80 (78 + 2 written `pp`), `PE-dominant` 3, `Uncertain` 6.
* **Analysis set: 178 flakes (98 PE / 80 PP).** Excluded: 3 `PE-dominant` and 5 `Uncertain` (all white Bopp shred; libraries say PVC / mineral) — kept aside as a case study; `C11_Q_Plastic_Bopp_A5` has no spectrum; 4 extra rescans (`D3.1`, `D4.1`, `Sink_PP_C5.1`, `Bopp_A.0`) are unused.
* Spectra: Bruker OPUS exports, 6845 points, 399-6999 cm⁻¹, 0.964 cm⁻¹ step, identical grid for all files; the native `.0` binary was verified identical to `.dpt` (max diff 5e-6).
* Colorimeter: CIELAB L*a*b* (D65, 10°) per flake, same order as the GT sheet (checked).
* Black flakes: 22 (17 PE – all Circular 11; 5 PP – all Impact, plate position A1). **This is a source confound** (below).

Figures: `01` mean spectra, `02` black vs others, `03` colour space, `04` class by source.

## 3. Evaluation protocol (what "split" means here)

One spectrum per physical flake (no replicate leakage). Three complementary protocols, all stratified by class × data family × black:

| Protocol | Purpose |
|---|---|
| Repeated stratified 5-fold CV ×5 | stable estimate (all 178 flakes) |
| Fixed 75/25 hold-out (`results/splits/holdout_split.csv`) | simple train/validate split (133 / 45) |
| **Leave-one-source-out** (9 batches/plates) | generalisation to an unseen batch — closest to deployment |
| Extra: family shift (Impact ↔ Circular 11); train non-black → test black; CV inside the black flakes | domain-shift and black-polymer questions |

Hyper-parameters are tuned only inside training folds (grouped / augmentation-aware inner CV).

## 4. Experiments and results

### 4.1 Preprocessing ablation (Fig. 00, `tables/exp1_preprocessing.csv`)
5 regions × 7 preprocessing chains × 3 models: **every configuration lands at 98.9-99.7 %** (0.6-2 errors of 178, i.e. the one blend flake plus at most one fold flip). Chains without a normalisation step are marginally worse (98.9-99.4 %). Baseline correction (poly/ALS), SG derivatives and SNV are interchangeable for accuracy; they matter for *robustness* (4.4). Chosen: SG 1st derivative (window 11) + SNV on 600-1800 ∪ 2700-3100 cm⁻¹.

### 4.2 Model benchmark (Fig. 06, 07; `tables/benchmark_all_models.csv`)
Ten classical models, four deep backbones (MLP, 1-D CNN, ResNet-1D, patch-Transformer; with/without augmentation), colour-only models and fusions:

| Group | CV acc | Hold-out | Leave-source-out |
|---|---|---|---|
| FTIR: logreg, lin-SVM, RBF-SVM, PLS-DA, PCA-LDA, RF, ExtraTrees | 99.4 % | 100 % | 99.4 % |
| FTIR: kNN | 99.4 % | 97.8 % | 99.4 % |
| FTIR: LightGBM / XGBoost | 99.1 / 99.0 % | 95.6 % | 99.4 / 98.3 % |
| Deep: CNN, ResNet (±aug) | 99.4 % | 100 % | 99.4 % |
| Deep: MLP, Transformer | 99.4 % | 97.8 % | 99.4 % |
| **Colour only** (logreg / RBF-SVM / RF) | 58.9 / 69.8 / 74.6 % | 48.9 / 48.9 / 64.4 % | 55.1 / 59.6 / 73.0 % |
| FTIR+colour early / late fusion | 99.4 / 98.9-99.4 % | 95.6-100 % | 99.4 % |

All FTIR models make the same single error (the blend). Boosted trees are the weakest on small data. Backbone choice is therefore decided by robustness and cost, not accuracy.

### 4.3 Does colour help — especially for black polymers? (Fig. 03, 08; `tables/exp4_shift_black.csv`)
* FTIR already classifies **all 22 black flakes correctly** (ATR spectra of black PE/PP are clean here; Fig. 02) — in CV, in leave-source-out, and when the model has **never seen a black flake** (train non-black → test black: 22/22).
* Colour alone is weak overall (59-75 %) and **does not transfer between batches**: trained on Impact → tested on Circular 11 it scores 10-43 %, i.e. *below chance*. A "predictor" that only knows the batch of origin scores 75.3 %, which is the level colour-only models reach — they are mostly learning *which batch a flake came from* (e.g. all black PP are Impact, all black PE are Circular 11).
* Inside the 22 black flakes colour reaches 83-90 %, but this is exactly that confound, not evidence of colour discriminating polymers.
* Fusion (early or late) gives **no gain** (FTIR is already at ceiling) and is slightly worse under noise (Fig. 14: 95.2 % vs 98.6 % at σ = 10 % of peak).
* **Conclusion:** on this dataset colour is not needed and, with the present sampling, cannot be shown to help. A fair test of the black-polymer hypothesis needs black PP *and* black PE from the same batches, and genuinely hard black spectra (carbon-black-loaded, low-signal). The deployed model therefore ignores L*a*b* (the web form accepts it for traceability only).

### 4.4 Robustness to acquisition conditions (Fig. 09, 14; `tables/exp5_*.csv`, `exp8*.csv`)
Models are trained on clean spectra, then the test spectra are degraded. Measured instrument noise is σ = 1.9e-4 A (peak ≈ 0.32 A), so 2 % noise is ≈ 34× the real noise.

* Baseline tilt/offset: all models invariant (the derivative removes it).
* **Un-augmented RBF-SVM and PLS-DA collapse under wavenumber shift** (RBF-SVM 10.7 %, PLS-DA 19 % at ±8 cm⁻¹); logistic regression collapses under noise (58 % at 20 %) and blur.
* **Training-time augmentation** (noise, blur, gain, tilt, ±8 cm⁻¹ shift; group-aware tuning and calibration) fixes everything. The deployed **Augmented RBF-SVM keeps ≥ 99.2 % up to 5 % noise, 91 % at 20 % noise, 98.9 % at 32 cm⁻¹ blur, 99.4 % at 50× signal loss and 99.4 % at ±8 cm⁻¹ shift.**
* Ensembles: with an **un-augmented** member set they are worse than the single augmented model (noise 87 % at 10 %, blur 51 % at 32 cm⁻¹). Ensembles whose members are all augmented (AugSVM + AugLogReg + CNN) match AugSVM (noise 98.9 % at 10 %, 94.1 % at 20 %; blur 98.6 %), are marginally better at the extreme (20 % noise: 94.1 % vs 91.0 %) but 2.8× slower per spectrum (3.65 vs 1.33 ms) and 2.4× longer to train (34 s vs 14 s); it needs PyTorch at serving time. The simpler single model was kept; the ensemble is a drop-in option if extreme noise is expected.

### 4.5 Efficiency (Fig. 10, 11, 12, 15)
* **Little spectrum is needed.** The C-H stretch window alone (2800-3000 cm⁻¹) at ~15 cm⁻¹ spacing — **13 points** — gives 99.4 % leave-source-out; two narrow windows (PE doublet + PP 1376 band, **7 points**) also 99.4 %. Single-band windows are weaker: the 1376 cm⁻¹ band alone gives 92.7-96.6 % even at native resolution and 55-78 % with only 4 points; the PE doublet alone 80-93 % with ≤ 6 points.
* **An interpretable 2-feature rule works as well:** logistic regression on the band-height ratios A720/A1463 and A1376/A1463 → 99.4 % (CV and leave-source-out), AUC 1.00 (`tables/exp6c_band_ratio_rule.json`). The attribution plot (Fig. 12, logistic regression) is largely chemically sensible — positive (PP) weight near 2950 and in the PP fingerprint bands (≈ 840/810, 973/998 cm⁻¹), negative (PE) weight near the 720/730 doublet — but it also carries broad weights at 1500-1700 cm⁻¹ and 800-900 cm⁻¹ that are not specific bands (likely additives/oxidation/baseline), so it should be read as indicative only.
* **Little training data is needed.** 8 training flakes already give ≈ 100 % on the hold-out (Fig. 11) — good news for adding new sites/instruments, but it again reflects how easy the clean 2-class problem is.
* **Latency (Fig. 15, `tables/exp9_latency.csv`)**: every model infers in 1.0-1.8 ms/spectrum (ExtraTrees 12 ms); deployed model 1.33 ms median, p95 1.44 ms, 3.4 MB. Parsing a text `.dpt` costs ≈ 11 ms, a native OPUS `.0` ≈ 1.4 ms. The bottleneck of a real-time line is the spectrometer, not the model.

### 4.6 Previous model on this dataset (Fig. 13; `tables/exp7_*`)
The earlier Mix-plastic regressor (PLSR, HDPE % in HDPE/LDPE blends) was applied to our flakes via the repo's own code (bundle statically inspected first; the wavenumber axis had to be reconstructed — 498 points over 600-3000 cm⁻¹ — so this is approximate). **Its raw output is meaningless here: 100 % of predictions fall outside 0-100 % (PE ≈ 305, PP ≈ 626), so the nominal threshold gives 55 % accuracy.** As a *ranking score* it separates PE from PP almost perfectly (AUC 0.998; 99.4 % with a threshold fitted on the same data), which just shows the spectra are easily separable — it is not a usable classifier without recalibration, i.e. retraining. Hence our new classification model.

### 4.7 Out-of-distribution and blends
* A PCA reconstruction-error detector flags all 5 Bopp-shred spectra (novelty 14.2-14.4 vs threshold 1.67), which the libraries call PVC/mineral (GT: `Uncertain`). **The bare classifier calls them PP with 95-96 % probability — only the novelty check stops this being reported as a confident PP.** The service marks them `unreliable`. ("BOPP" normally denotes biaxially oriented *polypropylene* film, so it is worth checking whether the library IDs or the `Uncertain` label are what is off; the spectra are weak and filler-dominated, so we make no claim either way.)
* A "mixed PE+PP signature" review hint (both band ratios ≥ 0.78) flags 8/178 flakes (4.5 %) — all from Express Bags 13 and 17 — **7 of them verified PE** (branched PE also has a 1376 cm⁻¹ band) and the one CV error. It is a review hint, not a correction; the threshold was set post-hoc on this data.
* 3 `PE-dominant` flakes are classified as PE with high confidence (PP index ≤ 0.69); their GT status is genuinely ambiguous.

## 5. Deliverable: web service
`webapp/` (FastAPI + dependency-free single-page UI, works offline): drag-and-drop `.0` (native OPUS) / `.dpt` / `.csv` / `.txt`, batch upload, CSV export, per-spectrum probability, status (`ok` / `review` / `unreliable`), quality flags (weak signal, coverage, coarse sampling, novelty, mixed signature), spectrum plot with diagnostic bands, latency, optional L*a*b* (logged only). 11 automated tests (`tests/`). UI screenshots in `results/figures/ui/`.
Not yet done: authentication/HTTPS and public hosting (run behind your own reverse proxy), live-instrument integration (folder watcher), Dockerfile is untested.

## 6. Recommendations / next steps
1. **Validate on an independent ground truth** (DSC/TGA, or manual spectral review of the 9 "mixed" flakes) — the current labels are library-derived.
2. **Collect the hard cases:** black/dark flakes with low ATR signal, PP/PE blends, filled and contaminated flakes, other polymers (PS, PET, PVC…), and a second instrument — the real test of the black-polymer question, and of colour fusion.
3. Add an "other polymer / reject" class (the novelty detector is a stop-gap).
4. Keep the model FTIR-only until colour is shown to help on batch-balanced black data.
5. If hardware cost matters, a ~7-point filter/low-resolution sensor looks viable on this data (Fig. 10) — worth a feasibility test.
