# PP / PE FTIR classifier - portable demo

Runs the web service on your own computer (nothing is sent anywhere). Model: augmented RBF-SVM, FTIR only (CV 99.4 %, hold-out 45/45).

## Run (macOS / Linux)
1. Unzip, open Terminal in this folder (or double-click `run_demo.command`; if macOS blocks it: right-click -> Open, or `chmod +x run_demo.command`).
2. `./run_demo.command` - first run creates `.venv` and installs the packages (needs Python 3.10-3.13; `brew install python@3.12` if missing).
3. The browser opens http://127.0.0.1:8000. Click an example on the left, or drop `.0` / `.dpt` / `.csv` spectra on the page.
   Other port: `PORT=8010 ./run_demo.command`.

Open the page from that address (not as a saved file): a red banner at the top tells you if the service cannot be reached.

## Notes for the demo
* The model was trained on all 178 labelled flakes, so the bundled examples are training data. The honest accuracy numbers are in the report (cross-validation, hold-out, leave-one-source-out).
* L*a*b* colour boxes are accepted but ignored by the model.
* `models/pppe_ftir_model.joblib` is a pickle - only use bundles you trained yourself.
* No login / HTTPS: keep it on 127.0.0.1 (the default).
