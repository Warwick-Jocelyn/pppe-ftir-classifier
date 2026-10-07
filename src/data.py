"""Dataset loading for the PP-vs-PE FTIR classification work package.

Labels: the LAST column of `FTIR Classification_GT_Labels.xlsx` ("Varified Polymer ID") is the
ground truth. Folder names (e.g. "Float PE") are NOT reliable - many "Float PE" flakes are PP.

Spectra: Bruker OPUS exports (*.dpt, two whitespace-separated columns: wavenumber, absorbance),
one per flake, on a common 6845-point grid (6998.8 -> 399.2 cm-1, step 0.964).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Dataset"
CACHE = ROOT / "results" / "cache"

LABEL_MAP = {"PE": "PE", "PP": "PP", "PE-DOMINANT": "PE-dominant", "UNCERTAIN": "Uncertain"}


def read_dpt(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column OPUS .dpt text export -> (wavenumber ascending, absorbance)."""
    arr = np.loadtxt(path)
    wn, a = arr[:, 0], arr[:, 1]
    order = np.argsort(wn)
    return wn[order], a[order]


def load_metadata() -> pd.DataFrame:
    gt = pd.read_excel(DATASET / "FTIR Classification_GT_Labels.xlsx", header=4)
    gt = gt.dropna(how="all").dropna(axis=1, how="all")
    last = gt.columns[-1]
    assert last.lower().startswith("varified"), f"unexpected last column {last!r}"

    col = pd.read_excel(DATASET / "Colorimeter_data.xlsx", header=4).dropna(how="all").dropna(axis=1, how="all")
    assert (col["Sample ID.1"].values == gt["Sample ID.1"].values).all(), "GT / colorimeter row order differs"

    df = pd.DataFrame(
        {
            "sample_id": gt["Sample ID.1"].astype(str).str.strip().values,
            "source": gt["Source"].astype(str).str.strip().values,
            "plate": gt["Plate location"].astype(str).str.strip().values,
            "colour_name": gt["Visual colour"].astype(str).str.strip().str.lower().values,
            "label_raw": gt[last].astype(str).str.strip().values,
            "pf_id": gt["FTIR Polymer ID"].astype(str).str.strip().str.upper().values,  # Polymerflow library ID
            "pf_score": gt["Polymerflow Match Score"].astype(float).values,
            "os_id": gt["FTIR Polymer ID.1"].astype(str).str.strip().str.upper().values,  # OpenSpecy library ID
            "os_score": gt["OpenSpecy  Match score"].astype(float).values,
            "L": col["Colorimenter L*"].astype(float).values,
            "a": col["Colorimenter a*"].astype(float).values,
            "b": col["Colorimenter b*"].astype(float).values,
        }
    )
    df["label"] = df["label_raw"].str.upper().map(LABEL_MAP)
    assert df["label"].notna().all(), df.loc[df["label"].isna(), "label_raw"].unique()
    # families used for source-held-out validation
    df["family"] = np.where(df["source"].str.startswith("Impact"), "Impact", "Circular11")
    df["is_black"] = df["colour_name"].eq("black")
    return df


def build_dataset(force: bool = False):
    """Return (meta_df, wn, X_raw) with one raw absorbance spectrum per labelled flake."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "raw_dataset.npz"
    meta = load_metadata()
    if f.exists() and not force:
        z = np.load(f, allow_pickle=True)
        keep = [s for s in meta["sample_id"] if s in set(z["ids"])]
        meta = meta.set_index("sample_id").loc[list(z["ids"])].reset_index()
        return meta, z["wn"], z["X"]

    files = {os.path.basename(p)[:-4]: p for p in glob.glob(str(DATASET / "**" / "*.dpt"), recursive=True)}
    rows, spectra, wn_ref = [], [], None
    for i, r in meta.iterrows():
        key = r["sample_id"] + ".0"
        if key not in files:
            print(f"[data] no spectrum for {r['sample_id']} -> dropped")
            continue
        wn, a = read_dpt(files[key])
        if wn_ref is None:
            wn_ref = wn
        assert len(wn) == len(wn_ref) and np.allclose(wn, wn_ref, atol=1e-3), f"grid differs: {key}"
        rows.append(i)
        spectra.append(a)
    meta = meta.loc[rows].reset_index(drop=True)
    X = np.vstack(spectra).astype(np.float64)
    np.savez_compressed(f, ids=meta["sample_id"].values, wn=wn_ref, X=X)
    return meta, wn_ref, X


if __name__ == "__main__":
    m, wn, X = build_dataset(force=True)
    print(m.shape, X.shape, wn[0], wn[-1])
    print(m["label"].value_counts())
