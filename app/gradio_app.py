from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from digit_robustness.modeling import predict_proba
from digit_robustness.vision import flatten_for_model, process_digit
ARTIFACTS = ROOT / "artifacts"


def _load_json(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_champion():
    selection = _load_json("selection.json")
    champion = selection.get("champion")
    family = selection.get("champion_family")
    if champion == "Logistic Regression":
        model = joblib.load(ARTIFACTS / "models" / "logistic_regression.joblib")
    elif champion == "Random Forest":
        model = joblib.load(ARTIFACTS / "models" / "random_forest.joblib")
    elif champion == "MLP Keras":
        model = tf.keras.models.load_model(ARTIFACTS / "models" / "mlp.keras")
    else:
        raise RuntimeError("Execute o pipeline FULL antes de iniciar o app.")
    return champion, family, model


def predict_external(image: np.ndarray | None):
    if image is None:
        return None, "Envie uma imagem.", pd.DataFrame(columns=["digito", "probabilidade"])
    champion, family, model = _load_champion()
    try:
        processed = process_digit(image)
    except ValueError as exc:
        return None, str(exc), pd.DataFrame(columns=["digito", "probabilidade"])
    x = flatten_for_model(processed, family)
    probabilities = predict_proba(model, x)[0]
    if family == "mlp":
        labels = np.arange(10)
    else:
        labels = np.asarray(model.classes_)
    prediction = int(labels[int(np.argmax(probabilities))])
    table = pd.DataFrame({
        "digito": labels.astype(int),
        "probabilidade": probabilities.astype(float),
    }).sort_values("probabilidade", ascending=False).reset_index(drop=True)
    text = f"Modelo: {champion} · Predição: {prediction} · Confiança: {probabilities.max():.2%}"
    return processed, text, table


def _benchmark_frame() -> pd.DataFrame:
    path = ROOT / "outputs" / "tables" / "benchmark_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["modelo", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
    return pd.read_csv(path)


def _overview_text() -> str:
    summary = _load_json("run_summary.json")
    selection = _load_json("selection.json")
    if not summary:
        return "Execute o pipeline FULL para preencher os resultados."
    return (
        f"### Champion selecionado na validação: {selection.get('champion', '—')}\n"
        f"- Split: 49.000 treino / 7.000 validação / 14.000 teste\n"
        f"- Status: {summary.get('status', '—')}\n"
        f"- Teste final usado somente após congelamento da seleção."
    )


def build_demo() -> gr.Blocks:
    ood = _load_json("ood_results.json")
    with gr.Blocks(title="Digit Robustness Lab") as demo:
        gr.Markdown("# Digit Robustness Lab\nBenchmark MNIST, robustez OOD e inferência externa.")
        with gr.Tab("Overview"):
            gr.Markdown(_overview_text())
            gr.Dataframe(value=_benchmark_frame(), interactive=False, label="Benchmark final")
        with gr.Tab("OOD Lab"):
            gr.JSON(value=ood, label="Diagnóstico ID × OOD")
            gr.Image(
                value=str(ROOT / "outputs" / "figures" / "ood_msp_id_vs_ood.png")
                if (ROOT / "outputs" / "figures" / "ood_msp_id_vs_ood.png").exists()
                else None,
                label="MSP: ID × OOD",
                interactive=False,
            )
        with gr.Tab("Digit Lab"):
            gr.Markdown("Envie uma imagem recortada contendo um único dígito. Aqui são aplicados tons de cinza, inversão automática, recorte e centralização. A folha completa de tinta azul usa a segmentação específica do script evaluate_handwritten.py. A confiança não é garantia de acerto.")
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Imagem original")
                processed_output = gr.Image(label="Processada 28×28", interactive=False)
            prediction_text = gr.Markdown()
            probability_table = gr.Dataframe(interactive=False, label="Probabilidades")
            button = gr.Button("Classificar")
            button.click(
                fn=predict_external,
                inputs=image_input,
                outputs=[processed_output, prediction_text, probability_table],
            )
    return demo


if __name__ == "__main__":
    build_demo().launch()
