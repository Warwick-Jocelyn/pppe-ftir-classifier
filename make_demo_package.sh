#!/bin/bash
# Build the portable demo package: deploy/pppe_ftir_demo/ and deploy/pppe_ftir_demo.zip (serving files only, ~4 MB).
set -e
cd "$(dirname "$0")"
OUT=deploy/pppe_ftir_demo
rm -rf deploy && mkdir -p $OUT/src $OUT/webapp/static $OUT/models
cp src/preprocess.py src/models.py src/inference.py $OUT/src/
cp webapp/app.py $OUT/webapp/ && cp webapp/static/index.html $OUT/webapp/static/ && cp -r webapp/examples $OUT/webapp/
cp models/pppe_ftir_model.joblib $OUT/models/
cp requirements-serve.txt run_demo.command $OUT/ && cp docs/DEMO_README.md $OUT/README.md
python3 - <<'PY'
import shutil
shutil.make_archive('deploy/pppe_ftir_demo', 'zip', 'deploy', 'pppe_ftir_demo')
print('built deploy/pppe_ftir_demo.zip')
PY
