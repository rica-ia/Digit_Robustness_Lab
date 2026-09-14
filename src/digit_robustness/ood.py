"""Class Masking e diagnósticos de incerteza OOD."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

DEFAULT_HIDDEN = (4, 7)


def known_classes(hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> np.ndarray:
    classes = np.array([d for d in range(10) if d not in set(hidden)], dtype=np.uint8)
    return classes


def restrict_training(
    x: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...] = DEFAULT_HIDDEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known = known_classes(hidden)
    mask = ~np.isin(y, hidden)
    if np.isin(y[mask], hidden).any():
        raise AssertionError("Classes ocultas permaneceram no treinamento.")
    mapping = {int(label): idx for idx, label in enumerate(known)}
    internal_y = np.asarray([mapping[int(label)] for label in y[mask]], dtype=np.int64)
    return x[mask], internal_y, known


def decode_internal(indices: np.ndarray, known: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size and (indices.min() < 0 or indices.max() >= len(known)):
        raise ValueError("Índice interno fora do mapa de classes.")
    return np.asarray(known)[indices]


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or np.any(p < 0):
        raise ValueError("Probabilidades devem formar matriz 2D não negativa.")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Linhas de probabilidade devem somar 1.")
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def uncertainty_summary(probabilities: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities)
    msp = p.max(axis=1)
    entropy = predictive_entropy(p)
    return {
        "msp_mean": float(np.mean(msp)),
        "msp_median": float(np.median(msp)),
        "msp_p90": float(np.quantile(msp, 0.90)),
        "msp_p95": float(np.quantile(msp, 0.95)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_median": float(np.median(entropy)),
    }


def hidden_vs_known_matrix(
    y_hidden: np.ndarray,
    pred_known: np.ndarray,
    hidden: tuple[int, ...],
    known: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(hidden), len(known)), dtype=int)
    hidden_map = {label: i for i, label in enumerate(hidden)}
    known_map = {int(label): j for j, label in enumerate(known)}
    for truth, pred in zip(y_hidden, pred_known):
        if int(truth) in hidden_map and int(pred) in known_map:
            matrix[hidden_map[int(truth)], known_map[int(pred)]] += 1
    return matrix


def plot_uncertainty_distributions(
    id_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_msp = id_probabilities.max(axis=1)
    ood_msp = ood_probabilities.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(id_msp, bins=bins, alpha=0.55, density=True, label="ID")
    ax.hist(ood_msp, bins=bins, alpha=0.55, density=True, label="OOD")
    ax.set_xlabel("Maximum Softmax Probability (MSP)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de confiança — ID vs OOD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
