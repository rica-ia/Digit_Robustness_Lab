"""Métricas, matrizes de confusão e análise de erros."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_10x10(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    if cm.shape != (10, 10):
        raise ValueError(f"Matriz esperada 10x10; obtida {cm.shape}")
    return cm


def top_confusions(cm: np.ndarray, top_n: int = 8) -> pd.DataFrame:
    """Ordena erros por contagem e calcula taxa relativa por classe real."""
    rows: list[dict[str, float | int]] = []
    totals = cm.sum(axis=1)
    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            if true_label == pred_label or cm[true_label, pred_label] == 0:
                continue
            count = int(cm[true_label, pred_label])
            rate = float(count / totals[true_label]) if totals[true_label] else 0.0
            rows.append(
                {
                    "real": true_label,
                    "predita": pred_label,
                    "casos": count,
                    "taxa_relativa": rate,
                }
            )
    frame = pd.DataFrame(rows, columns=["real", "predita", "casos", "taxa_relativa"])
    if frame.empty:
        return frame
    return frame.sort_values(
        ["casos", "taxa_relativa"], ascending=False
    ).head(top_n).reset_index(drop=True)


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: str | Path,
    x_labels: list[int] | np.ndarray | None = None,
    y_labels: list[int] | np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    x_labels = np.arange(cm.shape[1]) if x_labels is None else np.asarray(x_labels)
    y_labels = np.arange(cm.shape[0]) if y_labels is None else np.asarray(y_labels)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    frame.index.name = "modelo"
    return frame.reset_index()
