"""Pré-processamento de dígitos externos para aproximar o padrão MNIST."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def _to_grayscale_array(image: ImageLike) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            img = Image.fromarray(arr.astype(np.uint8)).convert("L")
        elif arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        else:
            raise ValueError("Imagem deve ser 2D ou RGB/RGBA.")
    return np.asarray(img, dtype=np.uint8)


def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Garante fundo escuro e traço claro, como no MNIST."""
    h, w = gray.shape
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if float(np.median(border)) > 127:
        return 255 - gray
    return gray.copy()


def _crop_foreground(img: np.ndarray) -> np.ndarray:
    peak = int(img.max())
    if peak < 10:
        raise ValueError("Imagem vazia ou sem traço detectável.")
    threshold = max(20, int(0.12 * peak))
    mask = img >= threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Nenhum traço foi detectado.")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def _resize_to_mnist_body(crop: np.ndarray, target: int = 20) -> np.ndarray:
    h, w = crop.shape
    if h <= 0 or w <= 0:
        raise ValueError("Bounding box inválida.")
    scale = target / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = Image.fromarray(crop, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def _center_on_canvas(body: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    h, w = body.shape
    y0 = (canvas_size - h) // 2
    x0 = (canvas_size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = body.astype(np.float32)
    cy, cx = center_of_mass(canvas)
    if not np.isfinite([cy, cx]).all():
        raise ValueError("Não foi possível calcular o centro de massa.")
    desired = (canvas_size - 1) / 2.0
    delta = (desired - cy, desired - cx)
    centered = shift(canvas, shift=delta, order=1, mode="constant", cval=0.0, prefilter=False)
    return np.clip(centered, 0, 255)


def process_digit(image: ImageLike) -> np.ndarray:
    """Retorna imagem 28x28 float32 normalizada para [0,1]."""
    gray = _to_grayscale_array(image)
    polarized = _normalize_polarity(gray)
    crop = _crop_foreground(polarized)
    body = _resize_to_mnist_body(crop, target=20)
    centered = _center_on_canvas(body, canvas_size=28)
    normalized = (centered / 255.0).astype(np.float32)
    if normalized.shape != (28, 28):
        raise AssertionError("Saída deve ser 28x28.")
    if normalized.min() < 0 or normalized.max() > 1:
        raise AssertionError("Saída deve estar em [0,1].")
    return normalized


def flatten_for_model(processed: np.ndarray, family: str) -> np.ndarray:
    family = family.lower()
    if family == "lr":
        return processed.reshape(1, 784).astype(np.float64)
    if family in {"rf", "mlp"}:
        return processed.reshape(1, 784).astype(np.float32)
    raise ValueError(f"Família desconhecida: {family}")
