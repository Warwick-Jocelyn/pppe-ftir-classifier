"""Build results/PPPE_FTIR_results_slides.pptx (python-pptx). All numbers come from results/tables (verified) or the report.

PE = blue and PP = orange are reserved for the two polymer classes everywhere in the deck.
"""
import re
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures" / "slides"
OUT = ROOT / "results" / "PPPE_FTIR_results_slides.pptx"

INK, NAVY, PE, PP, TEAL, GREY, LIGHT, MID = "1B2430", "0F1B2D", "1F6FB5", "D9620B", "2A9D8F", "8A94A6", "F3F5F8", "4A5568"
HEAD, BODY = "Cambria", "Calibri"
rgb = lambda h: RGBColor.from_string(h)

THEME_COLORS = dict(dk1=INK, lt1="FFFFFF", dk2=NAVY, lt2=LIGHT, accent1=PE, accent2=PP, accent3=TEAL, accent4="16324F", accent5=GREY, accent6="C0392B",
                    hlink=PE, folHlink=MID)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = 13.333, 7.5

# ---------------------------------------------------------------- theme (fonts + colours) so the deck restyles from one place
for part in prs.part.package.iter_parts():
    if str(part.partname).startswith("/ppt/theme/theme"):
        xml = part.blob.decode("utf8")
        for k, v in THEME_COLORS.items():
            xml = re.sub(rf"(<a:{k}>)(.*?)(</a:{k}>)", rf'\1<a:srgbClr val="{v}"/>\3', xml, flags=re.S)
        xml = re.sub(r'(<a:majorFont>\s*<a:latin typeface=")[^"]*', rf"\g<1>{HEAD}", xml)
        xml = re.sub(r'(<a:minorFont>\s*<a:latin typeface=")[^"]*', rf"\g<1>{BODY}", xml)
        xml = re.sub(r'(<a:theme [^>]*name=")[^"]*', r"\g<1>Polyflow PPPE", xml)
        part._blob = xml.encode("utf8")

LAYOUT_TITLE_ONLY = prs.slide_layouts[5]
LAYOUT_BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def tb(slide, x, y, w, h, paras, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None,
       space_after=0, line=None):
    """paras: str | list of str | list of list-of-runs; run = (text, {bold,color,size,italic,font})."""
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(paras, str): paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        if line: para.line_spacing = line
        runs = [(p, {})] if isinstance(p, str) else p
        for text, o in runs:
            r = para.add_run(); r.text = text; f = r.font
            f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color)); f.name = o.get("font", font)
    return s


def box(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, name=None, shadow=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = radius
    if fill is None: s.fill.background()
    else: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line: s.line.color.rgb = rgb(line); s.line.width = Pt(1.25)
    else: s.line.fill.background()
    if not shadow:
        sp = s._element.spPr
        from lxml import etree
        etree.SubElement(sp, "{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst")
    s.text_frame.text = ""
    return s


def title(slide, text, color=INK, size=30, y=0.42, h=1.0):
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(y), Inches(W - 1.2), Inches(h)
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT; p.line_spacing = 0.95
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb(color)
    return t


def footer(slide, n, dark=False):
    c = "A9B4C6" if dark else GREY
    tb(slide, 0.6, 7.08, 6, 0.25, "Polyflow · PP/PE FTIR classification", size=10, color=c, name="footer")
    tb(slide, W - 1.6, 7.08, 1.0, 0.25, str(n), size=10, color=c, align=PP_ALIGN.RIGHT, name="slide number")


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


_no = [1]


def nxt():
    _no[0] += 1
    return _no[0]


def content_slide(text, n=None, title_size=28):
    s = prs.slides.add_slide(LAYOUT_TITLE_ONLY)
    title(s, text, size=title_size)
    footer(s, nxt())
    return s


def dark_background(slide):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = rgb(NAVY)


def pic(slide, path, x, y, w=None, h=None, name=None):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if name: p.name = name
    return p


# ================================================================ 1. title
s = prs.slides.add_slide(prs.slide_layouts[0]); dark_background(s)
t = s.shapes.title; t.left, t.top, t.width, t.height = Inches(0.8), Inches(1.9), Inches(6.5), Inches(2.5)
tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT; r = p.add_run(); r.text = "PP vs PE flake identification from FTIR"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = rgb("FFFFFF")
sub = s.placeholders[1]; sub.left, sub.top, sub.width, sub.height = Inches(0.8), Inches(4.6), Inches(6.4), Inches(1.0)
sub.text_frame.word_wrap = True
for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(sub.text_frame, m, 0)
p = sub.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
r = p.add_run(); r.text = "Validation study, deployed model and web service"; r.font.size = Pt(20); r.font.color.rgb = rgb("CBD5E1")
tb(s, 0.8, 5.35, 6.4, 0.4, "Polyflow work package · results summary · 2 October 2026", size=14, color="94A3B8")
pic(s, FIG / "fig_title_spectra.png", 7.5, 1.7, w=5.3, name="mean PE and PP spectra")
tb(s, 7.5, 4.9, 5.3, 0.7, "Mean ATR-FTIR spectra of the 98 PE and 80 PP flakes: the two polymers differ in a handful of well-known bands.",
   size=12, color="94A3B8")
notes(s, "Task: classify plastic flakes as PP or PE from ATR-FTIR, test whether colorimeter data helps (especially on black polymers), "
         "and deliver a real-time web service. 178 verified flakes, 9 sources.")

# ================================================================ 2. dataset split
s = content_slide("Every test flake is unseen in training: the four protocols differ only in how we split", 2)
pic(s, FIG / "fig_split.png", 1.15, 1.42, w=11.0, name="dataset and split schematic")
cols = [("Test flakes are always unseen", "B, C: unseen flakes. D, E: unseen flakes from an unseen batch, the harder test."),
        ("No leakage", "One spectrum per flake; stratified splits; tuning and calibration only inside the training part."),
        ("Labels = verified last column", "Folder names are unreliable (‘Float PE’ is 16/24 PP). 8 ambiguous flakes and 1 without spectrum are excluded.")]
for i, (h, d) in enumerate(cols):
    x = 0.6 + i * 4.1
    tb(s, x, 5.95, 3.9, 0.3, h, size=15, bold=True, color=INK, name=f"takeaway {i+1} title")
    tb(s, x, 6.27, 3.9, 0.75, d, size=13, color=MID, name=f"takeaway {i+1} text")
notes(s, "Key message: in every protocol the flakes being scored were not used to train the model that scores them; the protocols differ only in how flakes are split. In 5-fold CV a new model is trained each round on ~142 flakes and tested on the other ~36, so every flake is tested exactly once. Panel A: 178 PE/PP flakes from 5 Impact plates and 4 Circular-11 batches. All 5 black PP are Impact; all 17 black PE are Circular 11 - a source confound. "
         "Protocols: repeated stratified 5-fold CV x5; fixed 133/45 hold-out; leave-one-source-out (9 folds); shift tests (Impact<->Circular 11, non-black -> black).")

# ================================================================ 3. pipeline
s = content_slide("One preprocessing chain, trained on degraded copies and calibrated: a single 3.4 MB model", 3)
steps = [("1", "Raw data", "OPUS .0 / .dpt\n6,845 points\n399–6,999 cm⁻¹"),
         ("2", "Crop", "600–1,800 and\n2,700–3,100 cm⁻¹\n→ 1,658 points"),
         ("3", "Derivative", "Savitzky–Golay\n1st derivative\n(window 11)"),
         ("4", "Normalise", "SNV per spectrum\n(gain and baseline\noffset removed)"),
         ("5", "RBF-SVM", "C, γ by grouped CV\non training data"),
         ("6", "Calibrate", "Platt scaling on\nout-of-fold scores\n→ P(PP)")]
bw, gap, x0, y0, bh = 1.78, 0.30, 0.6, 3.0, 1.85
for i, (n, ttl, d) in enumerate(steps):
    x = x0 + i * (bw + gap)
    hl = i >= 4
    box(s, x, y0, bw, bh, fill=("16324F" if hl else LIGHT), name=f"step {n} box")
    box(s, x + 0.14, y0 + 0.14, 0.34, 0.34, fill=(PP if hl else PE), shape=MSO_SHAPE.OVAL, name=f"step {n} badge")
    tb(s, x + 0.14, y0 + 0.14, 0.34, 0.34, n, size=13, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name=f"step {n} number")
    tb(s, x + 0.14, y0 + 0.6, bw - 0.28, 0.32, ttl, size=15, bold=True, color=("FFFFFF" if hl else INK), name=f"step {n} title")
    tb(s, x + 0.14, y0 + 0.98, bw - 0.24, 0.84, d.split("\n"), size=11, color=("D6DEEA" if hl else MID), name=f"step {n} text")
    if i < 5:
        box(s, x + bw + 0.04, y0 + bh / 2 - 0.12, gap - 0.08, 0.24, fill=GREY, shape=MSO_SHAPE.RIGHT_ARROW, name=f"arrow {n}")
# augmentation (training only), spanning steps 3-6, arrow into step 5
ax_ = x0 + 2 * (bw + gap); aw = 4 * bw + 3 * gap
box(s, ax_, 1.5, aw, 1.05, fill="FDEBDD", name="augmentation box")
tb(s, ax_ + 0.2, 1.6, aw - 0.4, 0.3, "TRAINING ONLY · 6 degraded copies per flake", size=12, bold=True, color="A8470A", name="augmentation title")
tb(s, ax_ + 0.2, 1.92, aw - 0.4, 0.6, "noise ≤ 4 % · blur ≤ 10 cm⁻¹ · gain 0.3–1.3× · baseline tilt · ±8 cm⁻¹ shift. Copies of one flake always stay in the same CV group.",
   size=11.5, color="7A3A0C", name="augmentation text")
box(s, x0 + 4 * (bw + gap) + bw / 2 - 0.13, 2.6, 0.26, 0.36, fill=PP, shape=MSO_SHAPE.DOWN_ARROW, name="augmentation arrow")
# quality / novelty branch (steps 1-3 width) and output (steps 4-6 width)
qw = 3 * bw + 2 * gap
box(s, x0, 5.15, qw, 1.35, fill="E8F4F2", name="QC box")
tb(s, x0 + 0.2, 5.24, qw - 0.4, 0.3, "AT INFERENCE · quality and novelty checks", size=12, bold=True, color="1B6F66", name="QC title")
tb(s, x0 + 0.2, 5.56, qw - 0.4, 0.9, ["PCA reconstruction error flags spectra unlike PE/PP (e.g. Bopp shred) → ‘unreliable’",
                                      "Weak C–H signal, missing range, coarse sampling or mixed PE+PP signature → ‘review’"],
   size=11, color="14534C", space_after=3, name="QC text")
ox = x0 + 3 * (bw + gap)
box(s, ox, 5.15, qw, 1.35, fill=NAVY, name="output box")
tb(s, ox + 0.25, 5.24, qw - 0.5, 0.3, "OUTPUT", size=12, bold=True, color="94A3B8", name="output label")
tb(s, ox + 0.25, 5.56, qw - 0.5, 0.9, [[("PE or PP + probability + status", {"bold": True, "size": 17, "color": "FFFFFF"})],
                                      [("1.3 ms per spectrum · web UI and CSV export", {"size": 12, "color": "A9B4C6"})]], space_after=4, name="output text")
tb(s, 0.6, 6.62, 12.1, 0.45, [[("Benchmarked on the same chain: ", {"bold": True, "color": INK}),
                              ("logistic regression, PLS-DA, linear/RBF SVM, PCA-LDA, kNN, random forest, ExtraTrees, XGBoost, LightGBM, MLP, 1-D CNN, ResNet-1D, Transformer; colour-only and FTIR + colour fusion.", {})]],
   size=12, color=MID, name="benchmarked models")
notes(s, "Preprocessing was chosen by an ablation of 5 regions x 7 chains: every configuration is within 98.9-99.7 %, so the choice is about robustness. "
         "Augmentation makes the RBF-SVM invariant to calibration shift (otherwise 11 % at 8 cm-1) and to noise/blur. Calibration and tuning use GroupKFold with groups = original flake so augmented copies never leak.")

# ================================================================ 3b. selected model + architectures
s = content_slide("The selected model is a small kernel SVM; the deep backbones we compared are tiny too")


def chain(slide, y, label, sub, boxes, hl=False, h=0.8):
    lx, cx0, right = 0.6, 3.15, W - 0.6
    box(slide, lx, y, 2.4, h, fill=("16324F" if hl else LIGHT), name=f"{label} label box")
    tb(slide, lx + 0.12, y + 0.07, 2.2, 0.3, label, size=13, bold=True, color=("FFFFFF" if hl else INK), name=f"{label} label")
    tb(slide, lx + 0.12, y + 0.4, 2.2, 0.38, sub, size=10, color=("C9D5E6" if hl else MID), name=f"{label} stats")
    n = len(boxes); ag = 0.26; bw_ = (right - cx0 - (n - 1) * ag) / n
    for i, (t_, d_) in enumerate(boxes):
        x = cx0 + i * (bw_ + ag)
        fill = "D6E6F5" if hl else "FFFFFF"
        b_ = box(slide, x, y, bw_, h, fill=fill, line=("16324F" if hl else "C5CCD8"), name=f"{label} box {i+1}")
        tb(slide, x + 0.08, y + 0.06, bw_ - 0.16, 0.26, t_, size=10.5, bold=True, color=INK, name=f"{label} box {i+1} title")
        tb(slide, x + 0.08, y + 0.31, bw_ - 0.16, h - 0.33, d_, size=9.5, color=MID, name=f"{label} box {i+1} text")
        if i < n - 1:
            box(slide, x + bw_ + 0.03, y + h / 2 - 0.1, ag - 0.06, 0.2, fill=GREY, shape=MSO_SHAPE.RIGHT_ARROW, name=f"{label} arrow {i+1}")


rows_ = [("Augmented RBF-SVM", "DEPLOYED · 1.3 ms · 3.4 MB",
          [("Input", "1,658-pt spectrum after SG derivative + SNV"), ("RBF-kernel SVM", "C = 1, γ = 1e-4; 260 support vectors (128 PE / 132 PP)"),
           ("Platt scaling", "fitted on out-of-fold scores of original + degraded copies"), ("Output", "P(PP), PE/PP call, quality status")], True),
         ("Linear baselines", "logistic regression · PLS-DA\n1.1 ms · < 0.2 MB",
          [("Input", "SNV spectrum, standardised"), ("Linear model", "L2 logistic regression, or PLS-DA with 2–12 components"), ("Output", "P(PP) (PLS-DA: thresholded score)")], False),
         ("1-D CNN", "23.6k params · 1.6 ms",
          [("Input", "829 points"), ("3 × conv block", "Conv k = 9 + BN + GELU + MaxPool; 16 → 32 → 64 ch"), ("Global pooling", "mean ‖ max → 128-d vector"), ("Dropout + Linear", "→ P(PP)")], False),
         ("ResNet-1D", "60.9k params · 1.8 ms",
          [("Input", "829 points"), ("Stem", "Conv k = 15, stride 2, 16 ch"), ("3 residual blocks", "16 → 32 → 64 ch (stride 2), k = 7"), ("Global avg pool", "Dropout + Linear → P(PP)")], False),
         ("Transformer", "ViT-style · 41.4k params · 1.5 ms",
          [("Input", "829 points"), ("Patch embedding", "16-pt patches → 48-d, CLS token, position emb."), ("2 × encoder layer", "4 heads, FFN 96, pre-norm"), ("CLS → LayerNorm", "Linear → P(PP)")], False)]
for i, (lab, sub_, bx, hl_) in enumerate(rows_):
    chain(s, 1.45 + i * 0.93, lab, sub_, bx, hl=hl_)
tb(s, 0.6, 6.15, 12.1, 0.8, [[("Training: ", {"bold": True, "color": INK}),
                              ("SVM and linear models on the preprocessed spectrum (SVM with 6 degraded copies per flake, ≈ 14 s incl. tuning). Deep nets: AdamW, one-cycle LR 2e-3, label smoothing 0.05, fresh augmentation every epoch, 60–80 epochs (≈ 13 s on CPU). ", {}),
                              ("Why the SVM: ", {"bold": True, "color": INK}), ("same accuracy as the deep nets, most robust to the stress tests, and no PyTorch needed at serving time.", {})]],
   size=11.5, color=MID, name="training caption")
notes(s, "All models consume the same preprocessed spectrum (600-1,800 and 2,700-3,100 cm-1, SG 1st derivative, SNV); deep nets use every 2nd point (829 points). "
         "Parameter counts and latencies from results/tables/exp9_latency.csv; the SVM hyper-parameters are read from the deployed bundle. Accuracy is identical across these models (slide on backbones), so the choice is about robustness and cost.")

# ================================================================ 3c. relation to Aru's code
s = content_slide("We reused Aru's design ideas, not his code: what we kept, extended, changed and dropped")
arows = [("", "Aru's repository (HDPE/LDPE)", "Our deployment (PP/PE)", "Status"),
         ("Task", "Regression: HDPE % in HDPE/LDPE mixtures", "Classification: PP vs PE with a probability", "Changed"),
         ("Preprocessing", "SNV → standard scaling; crop wavenumber range; keep every N-th point; fitted on training data only", "Same ideas (SNV, crop to two windows, optional decimation, fit on train only) + Savitzky–Golay derivative", "Kept + extended"),
         ("Models", "PLSR, PCR, SVR, Ridge, Lasso, RF, GBRT with a small grid search", "Same model-zoo + grid-search idea, as classifiers (PLS-DA, SVM, RF …) plus CNN / ResNet / Transformer", "Kept + extended"),
         ("Validation", "Grouped CV (replicates of one sample stay together); tuning on the training split only", "Kept; added leave-one-source-out, batch-shift and black-flake tests; groups = flake incl. its degraded copies", "Kept + extended"),
         ("Packaging", "ModelBundle: model + preprocessing + metadata in one .joblib", "One bundle: model + preprocessing + novelty detector + model card; plain importable classes", "Kept"),
         ("Fusion", "Late fusion: average of NIR and MIR models", "Tested FTIR + colour (early and late): no gain, so not deployed", "Tested"),
         ("Not taken", "Zero-pad / chop to force feature length; saved sampling step breaks --validate_only; HDPE-only target", "Interpolate onto the fixed native grid; own implementation", "Dropped")]
gt = s.shapes.add_table(len(arows), 4, Inches(0.6), Inches(1.45), Inches(12.1), Inches(0.58 * len(arows))); gt.name = "relation to Aru's code table"
tbl = gt.table
for j, wdt in enumerate([1.6, 4.5, 4.7, 1.3]): tbl.columns[j].width = Inches(wdt)
stcol = {"Changed": PP, "Kept + extended": "1B6F66", "Kept": "1B6F66", "Tested": MID, "Dropped": "A8470A"}
for i, row in enumerate(arows):
    tbl.rows[i].height = Inches(0.56)
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = ""; c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.margin_left = c.margin_right = Inches(0.1); c.margin_top = c.margin_bottom = Inches(0.03)
        p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = val; r.font.size = Pt(11); r.font.name = BODY
        c.fill.solid()
        if i == 0: c.fill.fore_color.rgb = rgb(NAVY); r.font.color.rgb = rgb("FFFFFF"); r.font.bold = True
        else:
            c.fill.fore_color.rgb = rgb("FFFFFF" if i % 2 else LIGHT); r.font.color.rgb = rgb(INK)
            if j == 0: r.font.bold = True
            if j == 3: r.font.bold = True; r.font.color.rgb = rgb(stcol[val])
tb(s, 0.6, 6.25, 12.1, 0.75, [[("Aru's shipped model on our flakes: ", {"bold": True, "color": INK}),
                               ("run through his own code it gave outputs outside 0–100 % for every flake (AUC 0.998 as a ranking score), so it could not be reused as a classifier. Our code is written from scratch; nothing was copied.", {})]],
   size=12, color=MID, name="aru caption")
notes(s, "Honest summary: the design (preprocess fitted on train, grouped CV, tune on train only, bundle everything in one joblib, model zoo + grid search) comes from reading Aru's repo; "
         "the implementation is new. His pad_or_chop would silently zero-pad a mismatched spectrum, which is why we resample onto the fixed grid instead.")

# ================================================================ 4. FTIR vs colour
s = content_slide("Colour does not help: FTIR alone is near-perfect, colour alone does not transfer between batches", 4)
cd = CategoryChartData()
cd.categories = ["Repeated CV", "Hold-out", "Leave-one-\nsource-out", "Train Impact →\ntest Circular 11", "Train Circular 11 →\ntest Impact", "Train non-black →\ntest black"]
cd.add_series("FTIR (deployed augmented SVM)", (99.4, 100.0, 99.4, 98.3, 100.0, 100.0))
cd.add_series("Colour L*a*b* only (random forest)", (74.6, 64.4, 73.0, 43.1, 42.5, 22.7))
cd.add_series("FTIR + colour (late fusion)", (99.0, 95.6, 99.4, 98.3, 99.2, 100.0))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.5), Inches(8.3), Inches(5.35), cd); gf.name = "accuracy by input type chart"
ch = gf.chart; ch.font.size = Pt(11); ch.font.name = BODY; ch.font.color.rgb = rgb(MID)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11)
ch.has_title = True; ch.chart_title.text_frame.text = "Accuracy (%) by validation protocol (n = 178 flakes)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13); ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(INK)
va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 100; va.major_unit = 20; va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = rgb("E3E7EE"); va.format.line.fill.background(); va.tick_labels.font.size = Pt(11)
ch.category_axis.tick_labels.font.size = Pt(10.5); ch.category_axis.format.line.color.rgb = rgb("C5CCD8")
plot = ch.plots[0]; plot.gap_width = 55; plot.overlap = -5
for ser, col in zip(plot.series, ["16324F", "B0A08A", TEAL]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
    ser.data_labels.show_value = True; ser.data_labels.number_format = "0"; ser.data_labels.number_format_is_linked = False
    ser.data_labels.font.size = Pt(9); ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
# conclusions
x = 9.15
tb(s, x, 1.55, 3.6, 0.35, "What the figure shows", size=14, bold=True, color=MID, name="conclusion header")
items = [("98–100 %", "FTIR in all six tests, incl. 22/22 black flakes it never saw in training"),
         ("10–43 %", "colour alone when trained on one batch family and tested on the other: below chance"),
         ("No gain", "from fusion; it is slightly worse (95.6 % hold-out). The deployed model ignores colour")]
for i, (big, d) in enumerate(items):
    y = 2.0 + i * 1.28
    tb(s, x, y, 3.6, 0.5, big, size=26, bold=True, color=[ "16324F", "8C6D3F", TEAL][i], font=HEAD, name=f"stat {i+1}")
    tb(s, x, y + 0.5, 3.6, 0.7, d, size=12.5, color=MID, name=f"stat {i+1} text")
box(s, x - 0.15, 5.95, 3.75, 0.9, fill=LIGHT, name="caveat box")
tb(s, x, 6.02, 3.45, 0.8, [[("Why: ", {"bold": True, "color": INK}), ("colour mostly encodes the batch. All black PP is from Impact, all black PE from Circular 11.", {})]], size=11.5, color=MID, name="caveat text")
notes(s, "FTIR: deployed augmented RBF-SVM. Colour-only: random forest on L*, a*, b*, chroma. Fusion: logistic regression on spectrum + L*a*b* stacked. "
         "A predictor that only knows the source scores 75.3 %, the level colour reaches. Within the 22 black flakes colour alone gets 83-90 %, but that is the same confound. "
         "A fair test of colour needs black PP and black PE from the same batches.")

# ================================================================ 5. backbones table
s = content_slide("The strong FTIR models tie on accuracy, so robustness and cost decide the choice", 5)
rows = [("Model", "CV", "Hold-out", "Leave-source-out", "Time (ms)"),
        ("Logistic regression", "99.4", "100", "99.4", "1.1"),
        ("PLS-DA", "99.4", "100", "99.4", "1.1"),
        ("RBF-SVM", "99.4", "100", "99.4", "1.1"),
        ("ExtraTrees", "99.4", "100", "99.4", "11.8"),
        ("LightGBM", "99.1", "95.6", "99.4", "1.3"),
        ("1-D CNN (augmented)", "99.4", "100", "99.4", "1.6"),
        ("ResNet-1D (augmented)", "99.4", "100", "99.4", "1.8"),
        ("Transformer (augmented)", "99.4", "97.8", "99.4", "1.5"),
        ("Augmented RBF-SVM (deployed)", "99.4", "100", "99.4", "1.3"),
        ("Colour-only random forest", "74.6", "64.4", "73.0", "–")]
gt = s.shapes.add_table(len(rows), 5, Inches(0.6), Inches(1.55), Inches(8.3), Inches(0.43 * len(rows))); gt.name = "backbone comparison table"
tbl = gt.table
for j, wdt in enumerate([3.3, 0.9, 1.3, 1.7, 1.1]): tbl.columns[j].width = Inches(wdt)
for i, row in enumerate(rows):
    tbl.rows[i].height = Inches(0.43)
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = ""; c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_left = c.margin_right = Inches(0.1)
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.add_run(); r.text = val; r.font.size = Pt(13); r.font.name = BODY
        deployed, colour = i == 9, i == 10
        c.fill.solid()
        if i == 0: c.fill.fore_color.rgb = rgb(NAVY); r.font.color.rgb = rgb("FFFFFF"); r.font.bold = True
        elif deployed: c.fill.fore_color.rgb = rgb("D6E6F5"); r.font.bold = True; r.font.color.rgb = rgb(INK)
        elif colour: c.fill.fore_color.rgb = rgb("EFE9E0"); r.font.italic = True; r.font.color.rgb = rgb("6B5B45")
        else: c.fill.fore_color.rgb = rgb("FFFFFF" if i % 2 else LIGHT); r.font.color.rgb = rgb(INK)
tb(s, 0.6, 6.55, 8.2, 0.4, "Accuracy in % (CV = repeated 5-fold), n = 178, hold-out n = 45. Time: one CPU thread, raw spectrum → probability. Augmented = trained with degraded copies.",
   size=11, color=GREY, name="table note")
x = 9.2
tb(s, x, 1.55, 3.6, 0.35, "What the table shows", size=14, bold=True, color=MID, name="conclusion header")
pts = [[("One error, every model: ", {"bold": True, "color": INK}), ("C11_Exp_Bag_13_A4, labelled PP but spectrally a PE/PP blend.", {})],
       [("Accuracy is saturated, ", {"bold": True, "color": INK}), ("so backbone differences (±1 flake) are not meaningful. Boosted trees are weakest on 178 flakes.", {})],
       [("Real-time is easy: ", {"bold": True, "color": INK}), ("every model needs 1–2 ms (ExtraTrees 12 ms); the spectrometer, not the model, limits throughput.", {})],
       [("So we chose by robustness ", {"bold": True, "color": INK}), ("→ next slide.", {})]]
tb(s, x, 2.0, 3.6, 4.6, pts, size=14, color=MID, space_after=12, name="conclusions")
notes(s, "Hold-out is 45 flakes, so one flake = 2.2 points; LightGBM 95.6 % = 2 errors. Wilson 95 % CI for the deployed model hold-out 45/45: 92.1-100 %. "
         "Not shown: PCA-LDA, kNN, linear SVM, RF, XGBoost, MLP (all in results/tables/benchmark_all_models.csv).")

# ================================================================ 6. robustness
s = content_slide("Training on degraded copies removes the weak spots; adding colour does not help", 6)
pic(s, FIG / "fig_robustness.png", 0.6, 1.4, w=12.1, name="robustness figure")
items = [("Without augmentation", "RBF-SVM falls to 11 % under a ±8 cm⁻¹ wavenumber shift; logistic regression to 58 % at 20 % noise and 46 % at 32 cm⁻¹ blur."),
         ("Deployed augmented model", "≥ 99 % up to 5 % noise, 98.9 % at 32 cm⁻¹ blur, 99.4 % at ±8 cm⁻¹ shift and with 50× less signal. 2 % noise ≈ 34× the measured detector noise."),
         ("Colour fusion", "No benefit and slightly worse with noisy spectra (95 % vs 99 % at 10 % noise)")]
for i, (h, d) in enumerate(items):
    xx = 0.6 + i * 4.1
    tb(s, xx, 5.4, 3.85, 0.3, h, size=15, bold=True, color=INK, name=f"takeaway {i+1} title")
    tb(s, xx, 5.72, 3.85, 1.2, d, size=13, color=MID, name=f"takeaway {i+1} text")
notes(s, "Models are trained on clean spectra and the held-out fold's spectra are degraded (CV, n=178). Measured detector noise sigma=1.9e-4 A vs peak 0.32 A. "
         "Un-augmented ensembles were also worse; ensembles of augmented members match the single augmented SVM but are 2.8x slower.")

# ================================================================ 6b. real time
s = content_slide("Is ~1 ms real-time? The classifier is not the limiting step; the measurement is")
pic(s, FIG / "fig_realtime.png", 0.5, 1.5, w=8.3, name="time budget figure")
x = 9.15
tb(s, x, 1.55, 3.6, 0.35, "What the figure shows", size=14, bold=True, color=MID, name="conclusion header")
pts = [[("1.3 ms ", {"bold": True, "color": "16324F"}), ("for the whole model chain (crop, derivative, SNV, SVM) on one CPU thread, no GPU: ≈ 750 spectra/s per thread, ≈ 4,000/s in batch.", {})],
       [("Reading the file ", {"bold": True, "color": "16324F"}), ("costs more than the model (1.4 ms native OPUS, 11.5 ms text .dpt); a whole web request is ≈ 10 ms.", {})],
       [("A typical FTIR measurement ", {"bold": True, "color": "16324F"}), ("takes far longer (order 0.1–10 s, instrument-dependent, not measured here), so the model never makes the line wait.", {})],
       [("Still open: ", {"bold": True, "color": "A8470A"}), ("whether this is fast enough for a sorting line depends on the target flakes/s and the spectrometer's scan time. We measured software only, not acquisition, transport or ejection.", {})]]
tb(s, x, 2.0, 3.6, 4.8, pts, size=12.5, color=MID, space_after=10, name="conclusions")
notes(s, "Latency was measured from the raw 6,845-point vector to the probability (median of 200 calls, one thread). The reference range for FTIR acquisition is a general order of magnitude, not a measurement from your instrument. "
         "Ask for the target throughput (flakes per second) and the scan settings to give a definite yes/no.")

# ================================================================ 7. efficiency + service
s = content_slide("Real time is not the bottleneck; the web service works", 7)
stats = [("1.3 ms", "per spectrum, model only (one CPU thread; ≈ 4,000 per s in batch)"),
         ("13 points", "of the C–H stretch band alone still give 99.4 % (leave-source-out)"),
         ("8 flakes", "are enough to reach ≈ 100 % on the hold-out (learning curve)"),
         ("2 ratios", "A720/A1463 and A1376/A1463 give 99.4 %: an interpretable fallback")]
for i, (big, d) in enumerate(stats):
    cx, cy = 0.6 + (i % 2) * 2.3, 1.5 + (i // 2) * 2.45
    box(s, cx, cy, 2.15, 2.25, fill=LIGHT, name=f"stat card {i+1}")
    tb(s, cx + 0.18, cy + 0.2, 1.85, 0.6, big, size=22, bold=True, color="16324F", font=HEAD, name=f"stat {i+1}")
    tb(s, cx + 0.18, cy + 0.9, 1.85, 1.3, d, size=12, color=MID, name=f"stat {i+1} text")
pic(s, FIG / "ui_black_flake_crop.png", 5.55, 1.45, w=7.2, name="web UI screenshot")
tb(s, 5.55, 6.3, 7.2, 0.75, [[("Web service: ", {"bold": True, "color": INK}), ("drag-and-drop native Bruker .0, .dpt, .csv or .txt; batch upload and CSV export; probability, status and quality flags; ", {}),
                             ("no login or HTTPS yet.", {"bold": True, "color": INK})],
                            [("Shown: a black PP flake (Impact), classified PP with 100 % probability.", {"italic": True, "color": GREY, "size": 11})]],
   size=11.5, color=MID, space_after=2, name="service caption")
notes(s, "Latency: median 1.33 ms, p95 1.44 ms (deployed model). Reading a text .dpt costs about 11 ms, native OPUS about 1.4 ms; a whole API call is about 10 ms. "
         "Minimal-spectrum and learning-curve results reflect how easy the clean two-class problem is, so treat them as feasibility indications for a cheaper sensor, not as guarantees.")

# ================================================================ 8. conclusions
s = prs.slides.add_slide(LAYOUT_TITLE_ONLY); dark_background(s)
title(s, "Conclusions", color="FFFFFF", size=34)
footer(s, nxt(), dark=True)
colw = 5.85
tb(s, 0.6, 1.5, colw, 0.4, "What the evidence supports", size=17, bold=True, color="7FD1C7", name="left header")
left = [[("FTIR alone separates PP from PE: ", {"bold": True, "color": "FFFFFF"}), ("99.4 % CV, 45/45 hold-out, 99.4 % on unseen batches, 22/22 black flakes.", {})],
        [("Colour adds nothing here ", {"bold": True, "color": "FFFFFF"}), ("and mostly encodes the batch; the deployed model is FTIR-only.", {})],
        [("Robust and fast: ", {"bold": True, "color": "FFFFFF"}), ("augmented RBF-SVM tolerates noise, blur and ±8 cm⁻¹ shift at 1.3 ms per spectrum.", {})],
        [("Delivered: ", {"bold": True, "color": "FFFFFF"}), ("trained model, validation report, and a tested web service for single files and batches.", {})]]
tb(s, 0.6, 2.0, colw, 3.8, left, size=16, color="CBD5E1", space_after=14, name="left bullets")
tb(s, 6.9, 1.5, colw, 0.4, "What it does not show yet", size=17, bold=True, color="FFB783", name="right header")
right = [[("Labels are library-derived: ", {"bold": True, "color": "FFFFFF"}), ("‘verified’ equals the Polymerflow ID on every flake. Needs independent truth (DSC / TGA).", {})],
         [("Small, easy dataset: ", {"bold": True, "color": "FFFFFF"}), ("178 flakes, saturated accuracy. The 100 % hold-out is not a blind test; a new batch is.", {})],
         [("Black polymers remain open: ", {"bold": True, "color": "FFFFFF"}), ("need black PP and PE from the same batches and low-signal, carbon-black flakes.", {})],
         [("Scope: ", {"bold": True, "color": "FFFFFF"}), ("PE vs PP only. Other polymers and blends need a reject class; the Bopp shred is flagged ‘unreliable’.", {})]]
tb(s, 6.9, 2.0, colw, 3.8, right, size=16, color="CBD5E1", space_after=14, name="right bullets")
box(s, 0.6, 6.1, W - 1.2, 0.8, fill="16324F", name="bottom line box")
tb(s, 0.85, 6.1, W - 1.7, 0.8, [[("Bottom line: ", {"bold": True, "color": "7FD1C7"}), ("use the FTIR-only model now; validate it on an independent, harder batch (black, filled, blended) before relying on it for those materials.", {"color": "FFFFFF"})]],
   size=16, anchor=MSO_ANCHOR.MIDDLE, name="bottom line")
notes(s, "Next steps: (1) independent ground truth for the 'mixed' flakes and a sample of the rest; (2) collect hard cases - black/dark, filled, blends, other polymers, a second instrument; "
         "(3) add a reject class; (4) keep FTIR-only until colour is shown to help on batch-balanced black data; (5) auth/HTTPS and instrument integration for deployment.")

prs.save(OUT)
print("saved", OUT)
