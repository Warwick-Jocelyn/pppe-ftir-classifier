"""Inference utilities shared by the web service and CLI tools.

parse_spectrum_bytes : robust reader for .dpt / .csv / .txt (two columns: wavenumber, signal)
Predictor            : loads the deployed bundle, preprocesses, predicts, adds quality flags
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import joblib
import numpy as np

from preprocess import NATIVE_WN, resample_to_native

N_SPEC = len(NATIVE_WN)
BANDS = {  # diagnostic polyolefin bands (cm-1)
    "PE rocking doublet 720/730": (700, 740),
    "PP CH3 umbrella 1376": (1360, 1392),
    "PP CH3 asym. stretch 2951": (2940, 2962),
    "CH2 stretch 2918/2849": (2830, 2935),
}


# Heuristic from the 178 verified flakes: both diagnostic ratios >= 0.78 flags 8 flakes (4.5 %), 7 of them verified PE
MIX_THR = 0.78


class SpectrumError(ValueError):
    pass


def parse_spectrum_bytes(raw: bytes, filename: str = "") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (wavenumber, absorbance, notes). Accepts whitespace/comma/semicolon/tab separated text."""
    notes: list[str] = []
    if raw[:4] == b"\x0a\x0a\xfe\xfe":  # Bruker OPUS magic number -> read the native binary directly
        import os
        import tempfile
        try:
            from brukeropus import read_opus
        except ImportError as e:  # pragma: no cover
            raise SpectrumError("Raw OPUS file received but 'brukeropus' is not installed; export as .dpt instead.") from e
        with tempfile.NamedTemporaryFile(suffix=".0", delete=False) as tmp:
            tmp.write(raw)
        try:
            f = read_opus(tmp.name)
            blk = f.a if hasattr(f, "a") and f.a is not None else None
            if blk is None:
                raise SpectrumError("OPUS file has no absorbance block (export absorbance, not transmittance/single-channel).")
            wn, a = np.asarray(blk.x, float), np.asarray(blk.y, float)
        except SpectrumError:
            raise
        except Exception as e:
            raise SpectrumError(f"Could not read OPUS file: {e}") from e
        finally:
            os.unlink(tmp.name)
        order = np.argsort(wn)
        notes.append("Read native Bruker OPUS binary (absorbance block).")
        return wn[order], a[order], notes
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p for p in re.split(r"[,\s;\t]+", line.strip()) if p]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue  # header or junk line
    if len(rows) < 50:
        raise SpectrumError("Could not find a two-column numeric spectrum (wavenumber, absorbance) in the file.")
    arr = np.array(rows)
    wn, a = arr[:, 0], arr[:, 1]
    if np.any(~np.isfinite(a)):
        raise SpectrumError("Spectrum contains NaN/inf values.")
    order = np.argsort(wn)
    wn, a = wn[order], a[order]
    if wn.max() < 100 and wn.min() > 1:  # micrometres?  (1 / (wn*1e-4)) - not supported, just flag
        raise SpectrumError("X axis does not look like wavenumber (cm-1).")
    if a.max() > 5 or (a.min() >= 0 and a.max() > 1.5):
        # %T style -> convert to absorbance
        T = np.clip(a / (100.0 if a.max() > 1.5 else 1.0), 1e-4, None)
        a = -np.log10(T)
        notes.append("Signal looked like transmittance; converted to absorbance with A = -log10(T).")
    return wn, a, notes


class Predictor:
    def __init__(self, bundle_path: str | Path):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["model"]
        self.info = self.bundle.get("info", {})
        self.thr = self.bundle.get("threshold", 0.5)
        self.ood = self.bundle.get("ood")

    # ---------------------------------------------------------------- quality control
    def quality(self, wn, a, x_native) -> list[dict]:
        flags = []
        lo, hi = float(wn.min()), float(wn.max())
        if lo > 700 or hi < 3000:
            flags.append({"level": "warn", "msg": f"Spectrum covers {lo:.0f}-{hi:.0f} cm-1; the model expects 700-3000 cm-1. "
                                                   "Missing regions are filled by edge-extension, accuracy may drop."})
        step = float(np.median(np.diff(wn)))
        if step > 4.5:
            flags.append({"level": "warn", "msg": f"Coarse sampling ({step:.1f} cm-1/pt); trained at ~1 cm-1 (accuracy robust to ~16 cm-1 blur)."})
        pk = float(x_native[(NATIVE_WN > 2800) & (NATIVE_WN < 3000)].max())
        if pk < 0.05:
            flags.append({"level": "bad", "msg": f"No significant C-H stretch band (peak {pk:.3f} A). Poor ATR contact, non-polyolefin or empty crystal - result unreliable."})
        elif pk < 0.12:
            flags.append({"level": "warn", "msg": f"Weak signal (C-H peak {pk:.3f} A); retake with better contact if possible."})
        pe_i, pp_i = self.band_indices(x_native)
        if min(pe_i, pp_i) >= MIX_THR:
            flags.append({"level": "warn", "msg": f"Mixed PE+PP spectral signature (720/1463 = {pe_i:.2f}, 1376/1463 = {pp_i:.2f}). "
                                                   "Could be a PE/PP blend or branched PE - manual review suggested."})
        if self.ood is not None:
            d = self.ood_score(x_native)
            if d > self.ood["threshold"]:
                flags.append({"level": "bad", "msg": f"Spectrum is unlike the PP/PE training spectra (novelty score {d:.2f} > {self.ood['threshold']:.2f}). "
                                                      "May be another polymer, filler-dominated, or a blend."})
        return flags

    @staticmethod
    def band_indices(x):
        def band(c, hw=6):
            return float(x[(NATIVE_WN > c - hw) & (NATIVE_WN < c + hw)].max())
        base = band(1463, 8) + 1e-9
        return band(720) / base, band(1376) / base

    def ood_score(self, x_native) -> float:
        o = self.ood
        z = o["pp"].transform(x_native[None])
        z = o["scaler"].transform(z)
        proj = o["pca"].transform(z)
        rec = o["pca"].inverse_transform(proj)
        return float(np.sqrt(((z - rec) ** 2).mean()))

    # ---------------------------------------------------------------- main API
    def predict(self, wn, a, lab=None, notes=None) -> dict:
        t0 = time.perf_counter()
        x = resample_to_native(wn, a)
        feats = np.concatenate([x, np.array(lab if lab is not None else [np.nan] * 3, float)])[None]
        use_colour = lab is not None and not np.any(np.isnan(lab))
        model = self.model if use_colour or "no_colour" not in self.bundle else self.bundle["no_colour"]
        if not use_colour and self.bundle.get("needs_colour", False):
            model = self.bundle["no_colour"]
        p = float(model.predict_proba(feats)[0, 1])
        dt = (time.perf_counter() - t0) * 1000
        label = "PP" if p >= self.thr else "PE"
        conf = max(p, 1 - p)
        flags = self.quality(wn, a, x)
        band = {}
        for nm, (lo, hi) in BANDS.items():
            m = (NATIVE_WN >= lo) & (NATIVE_WN <= hi)
            band[nm] = float(x[m].max())
        status = "ok"
        if any(f["level"] == "bad" for f in flags):
            status = "unreliable"
        elif conf < 0.9 or any(f["level"] == "warn" for f in flags):
            status = "review"
        return {"label": label, "p_pp": p, "p_pe": 1 - p, "confidence": conf, "status": status, "flags": flags,
                "notes": notes or [], "used_colour": bool(use_colour and self.info.get("colour_used", False)), "latency_ms": dt, "band_heights": band,
                "model": self.info.get("name", "model")}
