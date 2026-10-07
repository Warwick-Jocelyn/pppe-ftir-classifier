"""End-to-end tests for parsing + web API. Run: .venv/bin/python -m pytest tests -q"""
import io, sys, glob, os
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from inference import parse_spectrum_bytes, SpectrumError
from webapp.app import app

client = TestClient(app)
D = ROOT / 'Dataset'
def f(name):  # first match of a dpt/opus file by sample name
    return glob.glob(str(D / '**' / name), recursive=True)[0]

def test_parse_dpt_and_opus_agree():
    wd, ad, _ = parse_spectrum_bytes(open(f('Float_PP_A1.0.dpt'), 'rb').read())
    wo, ao, notes = parse_spectrum_bytes(open(f('Float_PP_A1.0'), 'rb').read())
    assert len(wd) == len(wo) == 6845 and np.allclose(ad, ao, atol=1e-4) and 'OPUS' in notes[0]

def test_parse_csv_with_header_and_commas():
    csv = 'wavenumber,absorbance\n' + '\n'.join(f'{w},{0.1+0.0001*w}' for w in np.linspace(500, 3500, 400))
    w, a, _ = parse_spectrum_bytes(csv.encode()); assert len(w) == 400

def test_parse_garbage_rejected():
    with pytest.raises(SpectrumError): parse_spectrum_bytes(b'hello world')

@pytest.mark.parametrize('name,truth', [('Float_PP_A1.0.dpt', 'PP'), ('C11_Aylesbury_PE_A1.0.dpt', 'PE'), ('Sink_PE_C3.0.dpt', 'PP')])
def test_api_predict_known(name, truth):
    r = client.post('/api/predict', files=[('files', (name, open(f(name), 'rb')))])
    assert r.status_code == 200
    res = r.json()['results'][0]
    assert res['label'] == truth, res
    assert res['latency_ms'] < 250 and 'plot' in res

def test_api_batch_csv_and_errors():
    files = [('files', ('a.dpt', open(f('Float_PP_A1.0.dpt'), 'rb'))), ('files', ('bad.txt', b'nonsense'))]
    r = client.post('/api/predict', files=files); out = r.json()['results']
    assert out[0]['label'] in ('PP', 'PE') and 'error' in out[1]
    r = client.post('/api/predict.csv', files=files); assert r.status_code == 200 and 'filename,prediction' in r.text

def test_api_colour_validation():
    r = client.post('/api/predict', files=[('files', ('a.dpt', open(f('Float_PP_A1.0.dpt'), 'rb')))], data={'L': '150', 'a': '1', 'b': '1'})
    assert r.status_code == 400

def test_flat_spectrum_flagged_unreliable():
    w = np.linspace(400, 4000, 3000); txt = '\n'.join(f'{x} {0.001*np.random.rand()}' for x in w)
    res = client.post('/api/predict', files=[('files', ('flat.txt', txt.encode()))]).json()['results'][0]
    assert res['status'] == 'unreliable'

def test_transmittance_autoconvert():
    w, a, _ = parse_spectrum_bytes(open(f('Float_PP_A1.0.dpt'), 'rb').read())
    T = 100 * 10 ** (-a)
    txt = '\n'.join(f'{x} {v}' for x, v in zip(w, T))
    res = client.post('/api/predict', files=[('files', ('T.txt', txt.encode()))]).json()['results'][0]
    assert res['label'] == 'PP' and any('transmittance' in n for n in res['notes'])

def test_static_and_health():
    assert client.get('/').status_code == 200 and client.get('/api/health').json()['status'] == 'ok'
