"""Local presentation API backed exclusively by the persisted experiment artifacts."""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from scipy.ndimage import rotate, gaussian_filter

APP_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DIGIT_PROJECT_ROOT', APP_DIR.parent)).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from digit_robustness.vision import process_digit, _to_grayscale_array, _normalize_polarity, _crop_foreground

MODELS = {'mlp': ('MLP Keras', 'mlp.keras'), 'rf': ('Random Forest', 'random_forest.joblib'), 'lr': ('Logistic Regression', 'logistic_regression.joblib')}
LOCK = Lock()
app = FastAPI(title='Digit Robustness Lab', docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=APP_DIR / 'static'), name='static')

def artifact(name):
    return json.loads((ROOT / 'artifacts' / name).read_text(encoding='utf-8'))

def png(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255, 0, 255).astype('uint8')
    stream = io.BytesIO()
    Image.fromarray(arr).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()

@lru_cache(maxsize=4)
def model(key):
    import joblib
    import tensorflow as tf
    filename = 'mlp_masked_4_7.keras' if key == 'masked' else MODELS[key][1]
    path = ROOT / 'artifacts' / 'models' / filename
    return tf.keras.models.load_model(path) if path.suffix == '.keras' else joblib.load(path)

def infer(pixels, key):
    from digit_robustness.modeling import predict_proba
    with LOCK:
        fitted = model(key)
        dtype = np.float64 if key == 'lr' else np.float32
        start = time.perf_counter()
        p = predict_proba(fitted, pixels.reshape(1, 784).astype(dtype))[0]
        elapsed = (time.perf_counter() - start) * 1000
        labels = np.load(ROOT / 'artifacts/models/masked_known_classes.npy') if key == 'masked' else getattr(fitted, 'classes_', np.arange(10))
    return {'key': key, 'name': 'MLP · classes 4 e 7 ocultas' if key == 'masked' else MODELS[key][0],
            'prediction': int(labels[p.argmax()]), 'confidence': float(p.max()), 'latency_ms': round(elapsed, 2),
            'entropy': float(-np.sum(p * np.log(np.maximum(p, 1e-12)))),
            'probabilities': [{'digit': int(label), 'value': float(value)} for label, value in zip(labels, p)]}

@lru_cache(maxsize=1)
def samples():
    with np.load(APP_DIR / 'samples.npz') as data:
        return data['images'].copy(), data['labels'].copy()

@app.get('/')
def home():
    return FileResponse(APP_DIR / 'static/index.html')

@app.get('/api/overview')
def overview():
    return {'benchmark': pd.read_csv(ROOT / 'outputs/tables/benchmark_metrics.csv').to_dict('records'),
            'selection': artifact('selection.json'), 'ood': artifact('ood_results.json'),
            'summary': artifact('run_summary.json'),
            'handwritten': pd.read_csv(ROOT / 'outputs/tables/handwritten_predictions.csv').to_dict('records')}

@app.get('/api/samples/{digit}')
def sample(digit: int, index: int = 0):
    if not 0 <= digit <= 9 or not 0 <= index < 100:
        raise HTTPException(422, 'Dígito ou índice fora do intervalo.')
    images, labels = samples()
    pixels = images[np.flatnonzero(labels == digit)[index]]
    return {'image': png(pixels), 'pixels': pixels.tolist(), 'digit': digit, 'index': index,
            'histogram': np.histogram(pixels, bins=16, range=(0, 256))[0].tolist()}

class Prediction(BaseModel):
    image: str | None = Field(None, max_length=8_000_000)
    digit: int | None = Field(None, ge=0, le=9)
    index: int = Field(0, ge=0, le=99)
    model: str = 'mlp'
    compare: bool = False
    rotation: float = Field(0, ge=-90, le=90)
    noise: float = Field(0, ge=0, le=0.8)
    blur: float = Field(0, ge=0, le=3)
    masked: bool = False

def prepare(req):
    if req.digit is not None:
        images, labels = samples()
        raw = images[np.flatnonzero(labels == req.digit)[req.index]]
        pixels = raw.astype(np.float32) / 255
        stages = [png(raw)] * 3 + [png(pixels)]
    else:
        if not req.image:
            raise ValueError('Desenhe, selecione um exemplo ou envie uma imagem antes de classificar.')
        try:
            data = base64.b64decode(req.image.split(',')[-1], validate=True)
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError('Use uma imagem com até 16 megapixels.')
                source.load()
                if source.mode in ('RGBA', 'LA') or 'transparency' in source.info:
                    rgba = source.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, 'white')
                    source = Image.alpha_composite(background, rgba).convert('RGB')
                else:
                    source = source.convert('RGB')
                source.thumbnail((1600, 1600))
                raw = np.asarray(source)
            gray = _normalize_polarity(_to_grayscale_array(raw))
            crop = _crop_foreground(gray)
            pixels = process_digit(raw)
            stages = [png(raw), png(gray), png(crop), png(pixels)]
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Arquivo inválido. Envie uma imagem PNG ou JPEG.') from exc
    original = pixels.copy()
    if req.rotation:
        pixels = rotate(pixels, req.rotation, reshape=False, order=1, mode='constant')
    if req.blur:
        pixels = gaussian_filter(pixels, req.blur)
    if req.noise:
        pixels = pixels + np.random.default_rng(42).normal(0, req.noise, pixels.shape)
    pixels = np.clip(pixels, 0, 1).astype(np.float32)
    return original, pixels, stages

@app.post('/api/predict')
def predict(req: Prediction):
    if req.model not in MODELS:
        raise HTTPException(422, 'Modelo desconhecido.')
    try:
        original, pixels, stages = prepare(req)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    keys = list(MODELS) if req.compare else [req.model]
    results = [infer(pixels, key) for key in keys]
    masked = infer(pixels, 'masked') if req.masked else None
    baseline = infer(original, req.model) if any([req.rotation, req.noise, req.blur]) else None
    return {'results': results, 'masked': masked, 'baseline': baseline, 'processed': png(pixels),
            'stages': stages, 'pixels': pixels.tolist(), 'expected': req.digit,
            'source': 'MNIST · teste' if req.digit is not None else 'Imagem externa',
            'perturbations': {'rotation': req.rotation, 'noise': req.noise, 'blur': req.blur}}

@app.get('/api/confusion/{key}')
def confusion(key: str):
    if key not in MODELS:
        raise HTTPException(404)
    suffix = {'mlp': 'mlp_keras', 'rf': 'random_forest', 'lr': 'logistic_regression'}[key]
    return {'matrix': pd.read_csv(ROOT / f'outputs/tables/confusion_{suffix}.csv', header=None).values.tolist()}

@lru_cache(maxsize=1)
def projection_data():
    from sklearn.decomposition import PCA
    images, labels = samples()
    pca = PCA(n_components=2, svd_solver='randomized', random_state=42)
    points = pca.fit_transform(images.reshape(len(images), -1).astype(np.float32) / 255)
    return {'points': points.tolist(), 'labels': labels.tolist(), 'variance': float(pca.explained_variance_ratio_.sum())}

@app.get('/api/projection')
def projection():
    return projection_data()

@app.get('/api/figures/{name}')
def figure(name: str):
    allowed = {'mlp_architecture.png', 'handwritten_detection.jpg', 'handwritten_processed_grid.png',
               'ood_msp_id_vs_ood.png', 'ood_hidden_vs_known.png', 'class_distribution.png'}
    if name not in allowed:
        raise HTTPException(404)
    return FileResponse(ROOT / 'outputs/figures' / name)

@app.get('/api/export/benchmark')
def export_benchmark():
    return FileResponse(ROOT / 'outputs/tables/benchmark_metrics.csv', filename='benchmark_mnist.csv', media_type='text/csv')

@app.get('/api/export/report')
def export_report():
    return overview()

class PerformanceRequest(BaseModel):
    model: str = 'all'
    count: int = Field(100, ge=10, le=1000)
    batch_size: int = Field(32, ge=1, le=256)
    repeats: int = Field(3, ge=1, le=5)

@app.post('/api/performance')
def performance(req: PerformanceRequest):
    from digit_robustness.modeling import predict_proba
    if req.model not in (*MODELS, 'all'):
        raise HTTPException(422, 'Modelo desconhecido.')
    images, labels = samples()
    # Balanced round-robin over the ten classes, never first-N of class-sorted data.
    indices = np.arange(1000).reshape(10, 100).T.reshape(-1)[:req.count]
    raw, expected = images[indices], labels[indices]
    keys = list(MODELS) if req.model == 'all' else [req.model]
    rows = []
    for key in keys:
        with LOCK:
            fitted = model(key)
            x = raw.reshape(len(raw), 784).astype(np.float64 if key == 'lr' else np.float32) / 255
            predict_proba(fitted, x[:req.batch_size])  # Warm-up and loading excluded.
            latencies, batch_sizes, durations, accuracies, cpu_times = [], [], [], [], []
            for _ in range(req.repeats):
                predictions = []
                cpu_start = time.process_time()
                start = time.perf_counter()
                for offset in range(0, len(x), req.batch_size):
                    batch = x[offset:offset + req.batch_size]
                    tick = time.perf_counter()
                    p = predict_proba(fitted, batch)
                    latencies.append((time.perf_counter() - tick) * 1000)
                    batch_sizes.append(len(batch))
                    classes = getattr(fitted, 'classes_', np.arange(10))
                    predictions.extend(classes[p.argmax(axis=1)].tolist())
                durations.append(time.perf_counter() - start)
                cpu_times.append(time.process_time() - cpu_start)
                accuracies.append(float(np.mean(np.asarray(predictions) == expected)))
        wall = sum(durations)
        rows.append({'key': key, 'name': MODELS[key][0], 'accuracy': float(np.mean(accuracies)),
                     'throughput': req.count * req.repeats / wall,
                     'batch_p50_ms': float(np.percentile(latencies, 50)),
                     'batch_p95_ms': float(np.percentile(latencies, 95)),
                     'amortized_ms_per_image': wall * 1000 / (req.count * req.repeats),
                     'total_seconds': wall, 'cpu_seconds': sum(cpu_times),
                     'batch_latencies_ms': latencies, 'batch_sizes': batch_sizes,
                     'run_seconds': durations})
    import platform
    return {'results': rows, 'config': req.model_dump(), 'samples_per_class': np.bincount(expected, minlength=10).tolist(),
            'environment': {'python': platform.python_version(), 'system': platform.system(), 'processor': platform.machine()},
            'protocol': '1 aquecimento por modelo; carregamento e normalização excluídos. Latências p50/p95 por lote, incluindo o último lote parcial. Vazão inclui agregação das predições. Amostra do teste congelado, sem retreinamento.'}
