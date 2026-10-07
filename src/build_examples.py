"""Copy a handful of representative spectra into webapp/examples for the demo buttons."""
import sys, shutil, glob, os; sys.path.insert(0, 'src')
import pandas as pd
from data import load_metadata, DATASET
from evalkit import save_json
OUT = DATASET.parent / 'webapp' / 'examples'; OUT.mkdir(parents=True, exist_ok=True)
m = load_metadata().set_index('sample_id')
files = {os.path.basename(p)[:-4]: p for p in glob.glob(str(DATASET / '**' / '*.dpt'), recursive=True)}
picks = [
 ('Float_PP_A1',        'Black flake (Impact)'),
 ('C11_Aylesbury_PE_A1','Black flake (Circular 11)'),
 ('Float_PP_A6',        'White flake'),
 ('Sink_PE_C3',         'Coloured flake'),
 ('Mixed_PO_B2',        'Mixed-PO flake'),
 ('C11_Exp_Bag_13_A4',  'PE/PP blend (ambiguous)'),
 ('C11_Q_Plastic_Bopp_A2','Bopp shred (not PE/PP-clean)'),
]
idx = []
for sid, title in picks:
    key = sid + '.0'
    if key not in files: print('missing', sid); continue
    r = m.loc[sid]
    dst = OUT / f'{sid}.dpt'; shutil.copy(files[key], dst)
    idx.append(dict(file=dst.name, title=f'{sid}  ·  {title}', truth=r['label_raw'] + f" ({r['colour_name']})",
                    lab=[float(r['L']), float(r['a']), float(r['b'])]))
save_json(idx, OUT / 'index.json'); print(len(idx), 'examples')
