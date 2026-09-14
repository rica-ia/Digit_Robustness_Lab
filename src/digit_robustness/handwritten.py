"""Teste externo com folha de dígitos manuscritos em caneta esferográfica azul."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .modeling import predict_proba
from .vision import flatten_for_model, process_digit

MODEL_SPECS = {
    "Logistic Regression": ("lr", "logistic_regression.joblib"),
    "Random Forest": ("rf", "random_forest.joblib"),
    "MLP Keras": ("mlp", "mlp.keras"),
}


def load_rgb(image: str | Path | Image.Image) -> Image.Image:
    """Aplica orientação EXIF e converte a fotografia para RGB."""
    if isinstance(image, Image.Image):
        img = image.copy()
    else:
        img = Image.open(image)
    return ImageOps.exif_transpose(img).convert("RGB")


def blue_excess_mask(image: Image.Image, threshold: float = 10.0) -> np.ndarray:
    """Separa traços azuis do papel por excesso do canal B sobre R/G."""
    rgb = np.asarray(image, dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess = blue - (red + green) / 2.0
    return excess > float(threshold)


def segment_blue_digits(
    image: Image.Image,
    threshold: float = 10.0,
    min_area_ratio: float = 1e-4,
    padding: int = 30,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Detecta componentes de tinta azul e os ordena de cima para baixo/esquerda."""
    mask = blue_excess_mask(image, threshold=threshold)
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, _ = ndimage.label(mask, structure=structure)
    min_area = max(100, int(mask.size * min_area_ratio))
    h, w = mask.shape
    components: list[dict[str, Any]] = []
    for idx, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        local = labels[slc] == idx
        area = int(local.sum())
        if area < min_area:
            continue
        cy, cx = ndimage.center_of_mass(mask, labels, idx)
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        components.append({
            "label_id": idx,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
            "bbox": [x0, y0, x1, y1],
            "crop_bbox": [max(0, x0-padding), max(0, y0-padding), min(w, x1+padding), min(h, y1+padding)],
        })
    components.sort(key=lambda item: (item["cy"], item["cx"]))
    return mask, components


def prepare_component(mask: np.ndarray, component: dict[str, Any], dilation_iterations: int = 1) -> np.ndarray:
    x0, y0, x1, y1 = component["crop_bbox"]
    crop = mask[y0:y1, x0:x1]
    if dilation_iterations > 0:
        crop = ndimage.binary_dilation(crop, iterations=int(dilation_iterations))
    binary = crop.astype(np.uint8) * 255
    return process_digit(binary)


def _load_models(root: Path) -> dict[str, tuple[str, Any]]:
    models: dict[str, tuple[str, Any]] = {}
    model_dir = root / "artifacts" / "models"
    for name, (family, filename) in MODEL_SPECS.items():
        path = model_dir / filename
        if family == "mlp":
            model = tf.keras.models.load_model(path)
        else:
            model = joblib.load(path)
        models[name] = (family, model)
    return models


def _predict_one(model: Any, family: str, processed: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    x = flatten_for_model(processed, family)
    probabilities = predict_proba(model, x)[0]
    classes = np.arange(10) if family == "mlp" else np.asarray(model.classes_)
    index = int(np.argmax(probabilities))
    return int(classes[index]), float(probabilities[index]), probabilities, classes


def _annotated_detection(image: Image.Image, components: list[dict[str, Any]], expected: list[int], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for sample, (component, label) in enumerate(zip(components, expected), start=1):
        x0, y0, x1, y1 = component["bbox"]
        draw.rectangle((x0-18, y0-18, x1+18, y1+18), outline="red", width=5)
        draw.text((x0, max(0, y0-55)), f"#{sample} real={label}", fill="red", stroke_width=2, stroke_fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _plot_processed_grid(processed_images: list[np.ndarray], expected: list[int], path: Path) -> None:
    cols = 5
    rows = int(np.ceil(len(processed_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes):
        if idx < len(processed_images):
            ax.imshow(processed_images[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"#{idx+1} · real {expected[idx]}")
        ax.axis("off")
    fig.suptitle("Dígitos após segmentação da tinta azul e adequação ao padrão MNIST")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)



def _plot_champion_probabilities(probabilities: np.ndarray, expected: list[int], predicted: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(probabilities, aspect="auto", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Probabilidade")
    ax.set_xticks(np.arange(10), labels=np.arange(10))
    labels = [f"#{i+1} real={truth} pred={pred}" for i, (truth, pred) in enumerate(zip(expected, predicted))]
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Classe")
    ax.set_title("MLP Keras — distribuição de probabilidades nos dígitos manuscritos")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def evaluate_handwritten_sheet(
    root: str | Path,
    image_path: str | Path,
    expected_labels: list[int],
    threshold: float = 10.0,
    dilation_iterations: int = 1,
) -> dict[str, Any]:
    """Segmenta a folha, testa os três modelos e persiste evidências reproduzíveis."""
    root = Path(root).resolve()
    image_path = Path(image_path).resolve()
    image = load_rgb(image_path)
    mask, components = segment_blue_digits(image, threshold=threshold)
    if len(components) != len(expected_labels):
        raise ValueError(f"Esperados {len(expected_labels)} dígitos, detectados {len(components)}.")
    processed_images = [prepare_component(mask, c, dilation_iterations=dilation_iterations) for c in components]
    models = _load_models(root)
    prediction_rows: list[dict[str, Any]] = []
    probability_rows: dict[str, list[np.ndarray]] = {name: [] for name in models}
    for sample, (processed, truth) in enumerate(zip(processed_images, expected_labels), start=1):
        for name, (family, model) in models.items():
            pred, conf, probs, classes = _predict_one(model, family, processed)
            probability_rows[name].append(np.asarray(probs, dtype=float))
            prediction_rows.append({
                "sample": sample,
                "expected": int(truth),
                "model": name,
                "predicted": pred,
                "confidence": conf,
                "correct": bool(pred == truth),
            })
    predictions = pd.DataFrame(prediction_rows)
    summary = predictions.groupby("model", sort=False).agg(
        correct=("correct", "sum"),
        total=("correct", "size"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    summary["accuracy"] = summary["correct"] / summary["total"]

    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(tables / "handwritten_predictions.csv", index=False)
    _annotated_detection(image, components, expected_labels, figures / "handwritten_detection.jpg")
    _plot_processed_grid(processed_images, expected_labels, figures / "handwritten_processed_grid.png")

    selection_path = root / "artifacts" / "selection.json"
    champion_name = json.loads(selection_path.read_text(encoding="utf-8"))["champion"]
    if champion_name not in models:
        raise ValueError(f"Campeão desconhecido: {champion_name}")
    mlp_probs = np.vstack(probability_rows["MLP Keras"])
    mlp_pred = predictions[predictions["model"] == "MLP Keras"].sort_values("sample")["predicted"].astype(int).tolist()
    champion_rows = predictions[predictions["model"] == champion_name].sort_values("sample")
    champion_pred = champion_rows["predicted"].astype(int).tolist()
    _plot_champion_probabilities(
        mlp_probs,
        expected_labels,
        mlp_pred,
        figures / "handwritten_mlp_probabilities.png",
    )

    result = {
        "status": "PASS",
        "file": image_path.name,
        "ink": "caneta esferografica azul",
        "segmentation": {
            "detected_digits": len(components),
            "expected_digits": len(expected_labels),
            "blue_excess_threshold": float(threshold),
            "dilation_iterations": int(dilation_iterations),
            "ordering": "top-to-bottom_then_left-to-right",
        },
        "expected_labels": [int(v) for v in expected_labels],
        "model_summary": {
            row["model"]: {
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
                "mean_confidence": float(row["mean_confidence"]),
            }
            for _, row in summary.iterrows()
        },
        "champion": champion_name,
        "champion_predictions": champion_pred,
        "champion_accuracy": float((np.asarray(champion_pred) == np.asarray(expected_labels)).mean()),
    }
    return result
