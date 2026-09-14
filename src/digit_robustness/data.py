"""Carregamento, validação e divisão estratificada do MNIST."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

RANDOM_STATE = 42


def load_mnist_flat() -> tuple[np.ndarray, np.ndarray]:
    """Carrega as 70 mil imagens MNIST como uint8 e vetoriza 28x28 -> 784."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    x = x.reshape(len(x), -1)
    validate_raw_mnist(x, y)
    return x.astype(np.uint8, copy=False), y.astype(np.uint8, copy=False)


def validate_raw_mnist(x: np.ndarray, y: np.ndarray) -> None:
    """Falha cedo se formato, classes ou escala do MNIST forem inesperados."""
    if x.shape != (70000, 784) or y.shape != (70000,):
        raise ValueError(f"Shape inesperado: X={x.shape}, y={y.shape}")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Pixels devem ser numéricos.")
    if float(x.min()) < 0 or float(x.max()) > 255:
        raise ValueError("Pixels fora do intervalo 0..255.")
    if not np.all(x == np.floor(x)):
        raise ValueError("MNIST cru deve conter intensidades inteiras.")
    if not np.array_equal(np.unique(y), np.arange(10)):
        raise ValueError("Rótulos esperados: dígitos 0..9.")


def make_split_indices(y: np.ndarray, random_state: int = RANDOM_STATE) -> Dict[str, np.ndarray]:
    """Cria split exato 70/10/20 com estratificação em ambas as divisões."""
    all_idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=0.125,  # 7.000 de 56.000 = 10% do total
        stratify=y[dev_idx],
        random_state=random_state,
    )
    splits = {"train": train_idx, "validation": val_idx, "test": test_idx}
    validate_split_indices(y, splits)
    return splits


def validate_split_indices(y: np.ndarray, splits: Dict[str, np.ndarray]) -> None:
    expected = {"train": 49000, "validation": 7000, "test": 14000}
    for name, size in expected.items():
        if len(splits[name]) != size:
            raise ValueError(f"{name}: esperado {size}, obtido {len(splits[name])}")
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if np.intersect1d(splits[left], splits[right]).size:
                raise ValueError(f"Sobreposição entre {left} e {right}.")
    union = np.concatenate([splits[n] for n in names])
    if len(np.unique(union)) != len(y):
        raise ValueError("O split não cobre exatamente toda a base.")
    for name, idx in splits.items():
        if not np.array_equal(np.unique(y[idx]), np.arange(10)):
            raise ValueError(f"{name} não contém todas as classes.")


def save_split_indices(splits: Dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **splits)


def normalize_for_family(x_raw: np.ndarray, family: str) -> np.ndarray:
    """Normaliza [0,255] -> [0,1] preservando dtype definido por família."""
    family = family.lower()
    if family == "lr":
        dtype = np.float64
    elif family in {"rf", "mlp"}:
        dtype = np.float32
    else:
        raise ValueError(f"Família desconhecida: {family}")
    x = x_raw.astype(dtype, copy=False) / dtype(255.0)
    if x.min() < 0 or x.max() > 1:
        raise ValueError("Normalização fora de [0,1].")
    return x
