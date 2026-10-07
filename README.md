# PP / PE FTIR classification (Polyflow)

## Quick start: run the web demo on your own computer (no server needed)
```bash
git clone <your-repo-url> && cd <repo>      # or unzip deploy/pppe_ftir_demo.zip
./run_demo.command                          # macOS: double-click works too. Needs Python 3.10-3.13
```
Opens http://127.0.0.1:8000 (open it from that address, not as a saved file). First run installs `requirements-serve.txt` (~1-2 min).
GitHub only stores the code; the site runs on the computer where you start it.
The raw data (`Dataset/`) is not in the repo. To rebuild the portable zip after changing code or the model: `./make_demo_package.sh`.


Real-time PP-vs-PE flake identification from ATR-FTIR spectra, with a web service and a full validation study.
**Read `results/REPORT.md` first** (results, caveats, recommendations).

```
Dataset/            input data (GT labels xlsx - LAST column is the truth, colorimeter xlsx, OPUS .0 / .dpt spectra)
src/                data loading, preprocessing, models, experiments, final training, inference
webapp/             FastAPI service + single-page UI (+ demo examples)
models/             deployed model bundle (pppe_ftir_model.joblib)
results/            REPORT.md, figures/, tables/, splits/, cache/ (logs, OOF predictions)
tests/              pytest suite (parsing + API)
```

## Run the web service
```bash
cd /home/supervisor/Jocelyn/Polyflow/PPPE_Classification_FTIR
./scripts_serve.sh                 # http://127.0.0.1:8000   (./scripts_serve.sh 0.0.0.0 8000 to expose on the LAN)
.venv/bin/python -m pytest tests -q
```
API: `POST /api/predict` (multipart `files[]`, optional `L`,`a`,`b`) → JSON; `POST /api/predict.csv`; `GET /api/model`, `/api/health`.
Accepts native Bruker `.0`, `.dpt`, `.csv`, `.txt` (any wavenumber grid; %T is auto-converted). `?demo=N` in the URL auto-runs example N.
No authentication / HTTPS is built in - put it behind a reverse proxy before exposing it beyond a trusted network. `Dockerfile` provided (untested).

## Reproduce the study
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/data.py                      # build cache from Dataset/
.venv/bin/python src/eda.py                       # EDA figures
.venv/bin/python src/exp1_preprocessing.py        # preprocessing ablation
.venv/bin/python src/exp2_benchmark.py            # classical + colour + fusion, 3 protocols
.venv/bin/python src/exp3_deep.py                 # MLP / CNN / ResNet / Transformer
.venv/bin/python src/exp4_shift_black.py          # source shift + black-flake experiments
.venv/bin/python src/exp5_robustness.py           # stress tests (CANDIDATES=1|final for the deployment candidates)
.venv/bin/python src/exp6_efficiency.py           # minimal spectrum / learning curve / 2-ratio rule
.venv/bin/python src/exp7_old_model.py            # previous HDPE/LDPE regressor on this data
.venv/bin/python src/exp9_latency.py              # latency
.venv/bin/python src/exp10_final_eval.py          # deployed-model evaluation
.venv/bin/python src/train_final.py               # fit on all 178 flakes -> models/pppe_ftir_model.joblib
.venv/bin/python src/make_report.py               # figures
.venv/bin/python src/build_processed_dataset.py   # Dataset/Processed (train/val/test copy)
.venv/bin/python src/verify_processed_dataset.py  # independent check of Dataset/Processed
.venv/bin/python src/make_slide_figures.py        # figures for the slide deck
.venv/bin/python src/make_slides.py               # results/PPPE_FTIR_results_slides.pptx (python-pptx)
```

## Notes
* Labels: analysis uses PE/PP only (178 flakes). `PE-dominant` and `Uncertain` are kept aside.
* The deployed model is FTIR-only; L*a*b* is accepted but ignored (see report §4.3).
* `models/pppe_ftir_model.joblib` is a pickle: only load bundles you trained yourself.
* Retrain after adding data: append spectra + labels to `Dataset/`, rerun `src/data.py` (force rebuild) and `src/train_final.py`.
