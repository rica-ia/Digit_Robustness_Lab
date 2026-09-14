"""Contract checks for the real model portal; no training or artifact mutations."""
import sys
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from web import app, samples, prepare, Prediction
client = TestClient(app)

def test_samples_are_balanced_and_native_pixels_are_preserved():
    images, labels = samples()
    assert images.shape == (1000, 28, 28)
    assert np.bincount(labels).tolist() == [100] * 10
    before, after, _ = prepare(Prediction(digit=7))
    expected = images[np.flatnonzero(labels == 7)[0]].astype(np.float32) / 255
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(after, expected)

@pytest.mark.parametrize('payload', [{}, {'image':'not-a-valid-image'}, {'model':'missing','digit':1}, {'digit':10}, {'digit':1,'noise':2}])
def test_invalid_predictions_are_rejected(payload):
    assert client.post('/api/predict', json=payload).status_code == 422

def test_reproducible_perturbation_is_bounded():
    req=Prediction(digit=8, rotation=25, noise=.15, blur=.8)
    a,b,_=prepare(req)
    _,c,_=prepare(req)
    assert b.shape==(28,28) and b.min()>=0 and b.max()<=1
    assert not np.array_equal(a,b)
    np.testing.assert_array_equal(b,c)

def test_real_model_probabilities_and_masked_class_labels():
    response=client.post('/api/predict',json={'digit':7,'compare':True,'masked':True})
    assert response.status_code==200
    data=response.json()
    assert len(data['results'])==3 and data['expected']==7
    for result in data['results']:
        assert abs(sum(p['value'] for p in result['probabilities'])-1)<1e-5
        assert 0<=result['prediction']<=9
    assert [p['digit'] for p in data['masked']['probabilities']]==[0,1,2,3,5,6,8,9]

def test_performance_uses_all_classes_and_accounts_for_partial_batch():
    response=client.post('/api/performance',json={'model':'lr','count':100,'batch_size':32,'repeats':2})
    assert response.status_code==200
    data=response.json(); result=data['results'][0]
    assert data['samples_per_class']==[10]*10
    assert result['batch_sizes']==[32,32,32,4]*2
    assert sum(result['batch_sizes'])==200
    assert result['throughput']==pytest.approx(200/result['total_seconds'])
    assert result['batch_p95_ms']>=result['batch_p50_ms']>0
    assert 0<=result['accuracy']<=1

def test_confusion_matrix_corresponds_to_full_test_size():
    for key in ['mlp','rf','lr']:
        matrix=np.asarray(client.get('/api/confusion/'+key).json()['matrix'])
        assert matrix.shape==(10,10) and matrix.sum()==14000

def test_invalid_performance_and_asset_access():
    assert client.post('/api/performance',json={'count':1001}).status_code==422
    assert client.get('/api/figures/selection.json').status_code==404
