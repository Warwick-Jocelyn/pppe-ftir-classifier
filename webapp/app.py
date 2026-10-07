"""PP / PE FTIR classification web service.

    uvicorn webapp.app:app --host 0.0.0.0 --port 8000

POST /api/predict    multipart: files[] (.0 OPUS / .dpt / .csv / .txt), optional L, a, b (CIELAB) -> JSON
GET  /api/examples   bundled example spectra (with verified labels, for demo)
GET  /api/model      deployed model card (+ benchmark numbers)
GET  /api/health
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import Predictor, SpectrumError, parse_spectrum_bytes  # noqa: E402

MODEL_PATH = ROOT / "models" / "pppe_ftir_model.joblib"
EXAMPLES = ROOT / "webapp" / "examples"
STATIC = ROOT / "webapp" / "static"
MAX_FILES = 200
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Polyflow PP/PE FTIR classifier", version="1.0")
_predictor: Predictor | None = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"model bundle not found at {MODEL_PATH}; run src/train_final.py")
        _predictor = Predictor(MODEL_PATH)
    return _predictor


def _plot_payload(wn: np.ndarray, a: np.ndarray, max_pts=900):
    m = (wn >= 600) & (wn <= 3200)
    x, y = wn[m], a[m]
    if len(x) > max_pts:
        idx = np.linspace(0, len(x) - 1, max_pts).astype(int)
        x, y = x[idx], y[idx]
    return {"x": np.round(x, 2).tolist(), "y": np.round(y, 5).tolist()}


def _parse_lab(L, a, b):
    vals = [L, a, b]
    if all(v in (None, "") for v in vals):
        return None
    try:
        arr = [float(v) for v in vals]
    except (TypeError, ValueError):
        raise HTTPException(400, "L*, a*, b* must all be numbers (or all left empty)")
    if not (0 <= arr[0] <= 100):
        raise HTTPException(400, "L* must be within 0-100")
    return arr


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None, "model_present": MODEL_PATH.exists()}


@app.get("/api/model")
def model_info():
    p = predictor()
    return {"info": p.info, "threshold": p.thr}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...), L: str | None = Form(None), a: str | None = Form(None),
                  b: str | None = Form(None), include_plot: bool = Form(True)):
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"max {MAX_FILES} files per request")
    pr = predictor()
    lab = _parse_lab(L, a, b)
    results = []
    for f in files:
        raw = await f.read()
        item = {"filename": f.filename}
        if len(raw) > MAX_BYTES:
            item["error"] = "file too large"
            results.append(item)
            continue
        try:
            wn, ab, notes = parse_spectrum_bytes(raw, f.filename or "")
            out = pr.predict(wn, ab, lab=lab, notes=notes)
            item.update(out)
            if include_plot:
                item["plot"] = _plot_payload(wn, ab)
        except SpectrumError as e:
            item["error"] = str(e)
        except Exception as e:  # never leak a stack trace to clients
            item["error"] = f"unexpected error while processing: {type(e).__name__}"
        results.append(item)
    return JSONResponse({"results": results})


@app.post("/api/predict.csv", response_class=PlainTextResponse)
async def predict_csv(files: list[UploadFile] = File(...), L: str | None = Form(None), a: str | None = Form(None),
                      b: str | None = Form(None)):
    res = json.loads((await predict(files, L, a, b, include_plot=False)).body)["results"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "prediction", "p_PP", "confidence", "status", "flags", "latency_ms", "error"])
    for r in res:
        w.writerow([r.get("filename"), r.get("label", ""), f"{r.get('p_pp', float('nan')):.4f}" if "p_pp" in r else "",
                    f"{r.get('confidence', float('nan')):.4f}" if "confidence" in r else "", r.get("status", ""),
                    " | ".join(f["msg"] for f in r.get("flags", [])), f"{r.get('latency_ms', float('nan')):.1f}" if "latency_ms" in r else "",
                    r.get("error", "")])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv")


@app.get("/api/examples")
def examples():
    idx = EXAMPLES / "index.json"
    return json.loads(idx.read_text()) if idx.exists() else []


@app.get("/api/example/{name}")
def example(name: str):
    p = (EXAMPLES / Path(name).name)
    if not p.exists() or p.suffix not in {".dpt", ".csv", ".txt"}:
        raise HTTPException(404, "no such example")
    return FileResponse(p, media_type="text/plain", filename=p.name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})
