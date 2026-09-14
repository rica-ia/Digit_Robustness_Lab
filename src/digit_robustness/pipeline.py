"""Orquestração reprodutível do experimento FULL."""
from __future__ import annotations

import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import load_mnist_flat, make_split_indices, normalize_for_family, save_split_indices
from .evaluation import classification_metrics, confusion_10x10, metrics_table, plot_confusion_matrix, top_confusions
from .handwritten import evaluate_handwritten_sheet
from .modeling import MLPConfig, build_logistic_regression, build_mlp, build_random_forest, fit_mlp, predict_labels, predict_proba
from .ood import DEFAULT_HIDDEN, decode_internal, hidden_vs_known_matrix, plot_uncertainty_distributions, restrict_training, uncertainty_summary
from .vision import flatten_for_model, process_digit

SEED = 42


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _acceptable(base: dict[str, float], candidate: dict[str, float]) -> bool:
    return candidate["accuracy"] >= base["accuracy"] and candidate["f1_weighted"] >= base["f1_weighted"]


def _eda_figures(x_raw: np.ndarray, y: np.ndarray, root: Path) -> None:
    figures = root / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    class_counts = pd.Series(y).value_counts().sort_index()
    class_counts.rename_axis("digito").reset_index(name="quantidade").to_csv(
        root / "outputs" / "tables" / "class_distribution.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(class_counts.index.astype(str), class_counts.values)
    ax.set_title("Distribuição das classes do MNIST")
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Quantidade")
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for digit, ax in enumerate(axes.flat):
        idx = int(np.flatnonzero(y == digit)[0])
        ax.imshow(x_raw[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Dígito {digit}")
        ax.axis("off")
    fig.suptitle("Exemplos do MNIST — uma imagem por classe")
    fig.tight_layout()
    fig.savefig(figures / "eda_grid_2x5.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fit_lr(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[Any, dict[str, Any]]:
    baseline = build_logistic_regression(C=1.0, solver="lbfgs")
    start = time.perf_counter()
    baseline.fit(x_train, y_train)
    baseline_seconds = time.perf_counter() - start
    baseline_metrics = classification_metrics(y_val, baseline.predict(x_val))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=build_logistic_regression(),
        param_grid={"C": [0.7, 1.0], "solver": ["lbfgs", "newton-cg"]},
        scoring="f1_weighted",
        cv=cv,
        n_jobs=2,
        refit=True,
        return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(x_train, y_train)
    search_seconds = time.perf_counter() - start
    candidate = search.best_estimator_
    candidate_metrics = classification_metrics(y_val, candidate.predict(x_val))
    accepted = _acceptable(baseline_metrics, candidate_metrics)
    final_model = candidate if accepted else baseline
    info = {
        "baseline": {"params": baseline.get_params(), "validation": baseline_metrics, "fit_seconds": baseline_seconds},
        "search_best_params": search.best_params_,
        "search_best_cv_f1_weighted": float(search.best_score_),
        "search_seconds": search_seconds,
        "candidate_validation": candidate_metrics,
        "candidate_accepted": accepted,
        "selected_params": final_model.get_params(),
        "selected_validation": candidate_metrics if accepted else baseline_metrics,
    }
    return final_model, info


def _fit_rf(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[Any, dict[str, Any]]:
    baseline = build_random_forest(n_estimators=160, max_depth=None)
    start = time.perf_counter()
    baseline.fit(x_train, y_train)
    baseline_seconds = time.perf_counter() - start
    baseline_metrics = classification_metrics(y_val, baseline.predict(x_val))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=build_random_forest(),
        param_grid={"n_estimators": [160, 240], "max_depth": [None, 28]},
        scoring="f1_weighted",
        cv=cv,
        n_jobs=2,
        refit=True,
        return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(x_train, y_train)
    search_seconds = time.perf_counter() - start
    candidate = search.best_estimator_
    candidate_metrics = classification_metrics(y_val, candidate.predict(x_val))
    accepted = _acceptable(baseline_metrics, candidate_metrics)
    final_model = candidate if accepted else baseline
    info = {
        "baseline": {"params": baseline.get_params(), "validation": baseline_metrics, "fit_seconds": baseline_seconds},
        "search_best_params": search.best_params_,
        "search_best_cv_f1_weighted": float(search.best_score_),
        "search_seconds": search_seconds,
        "candidate_validation": candidate_metrics,
        "candidate_accepted": accepted,
        "selected_params": final_model.get_params(),
        "selected_validation": candidate_metrics if accepted else baseline_metrics,
    }
    return final_model, info


def _fit_mlp_candidates(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[tf.keras.Model, MLPConfig, dict[str, Any]]:
    configs = [
        MLPConfig(128, 64, 1e-3, 128),
        MLPConfig(128, 64, 5e-4, 128),
        MLPConfig(192, 96, 1e-3, 128),
        MLPConfig(192, 96, 5e-4, 128),
    ]
    results: list[dict[str, Any]] = []
    models: list[tf.keras.Model] = []
    for idx, config in enumerate(configs):
        model = build_mlp(output_units=10, config=config, seed=SEED)
        history, fit_seconds = fit_mlp(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config=config,
            epochs=15,
            patience=3,
            verbose=0,
        )
        pred = predict_labels(model, x_val)
        metrics = classification_metrics(y_val, pred)
        results.append(
            {
                "config": config.__dict__,
                "validation": metrics,
                "fit_seconds": fit_seconds,
                "epochs_ran": len(history.history["loss"]),
                "best_val_loss": float(min(history.history["val_loss"])),
            }
        )
        models.append(model)
    baseline_metrics = results[0]["validation"]
    admissible = [0] + [i for i in range(1, len(results)) if _acceptable(baseline_metrics, results[i]["validation"])]
    selected_idx = max(
        admissible,
        key=lambda i: (
            results[i]["validation"]["f1_weighted"],
            results[i]["validation"]["accuracy"],
            -results[i]["fit_seconds"],
        ),
    )
    for i, row in enumerate(results):
        row["admissible_vs_baseline"] = i == 0 or _acceptable(baseline_metrics, row["validation"])
        row["selected"] = i == selected_idx
    return models[selected_idx], configs[selected_idx], {
        "baseline_validation": baseline_metrics,
        "candidates": results,
        "selected_config": configs[selected_idx].__dict__,
        "selected_validation": results[selected_idx]["validation"],
    }


def _predict_with_timing(model: Any, x: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    probabilities = predict_proba(model, x)
    if family == "mlp":
        predictions = probabilities.argmax(axis=1)
    else:
        predictions = np.asarray(model.classes_)[probabilities.argmax(axis=1)]
    elapsed = time.perf_counter() - start
    return predictions, probabilities, elapsed


def _save_environment(root: Path) -> None:
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "seed": SEED,
    }
    _json_dump(root / "environment_info.json", payload)


def run_full(root: str | Path, require_own_image: bool = False) -> dict[str, Any]:
    """Executa seleção na validação e avaliação final sem usar o teste para escolher modelos."""
    root = Path(root).resolve()
    for folder in [
        root / "artifacts" / "models",
        root / "outputs" / "figures",
        root / "outputs" / "tables",
        root / "data" / "own_digits",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    _save_environment(root)

    print("[1/8] Carregando MNIST e criando split estratificado 70/10/20...", flush=True)
    x_raw, y = load_mnist_flat()
    splits = make_split_indices(y, random_state=SEED)
    save_split_indices(splits, root / "artifacts" / "split_indices.npz")
    _eda_figures(x_raw, y, root)
    train_idx, val_idx, test_idx = splits["train"], splits["validation"], splits["test"]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    print("[2/8] Preparando representações por família...", flush=True)
    x_lr_train = normalize_for_family(x_raw[train_idx], "lr")
    x_lr_val = normalize_for_family(x_raw[val_idx], "lr")
    x_lr_test = normalize_for_family(x_raw[test_idx], "lr")
    x_f32_train = normalize_for_family(x_raw[train_idx], "mlp")
    x_f32_val = normalize_for_family(x_raw[val_idx], "mlp")
    x_f32_test = normalize_for_family(x_raw[test_idx], "mlp")

    print("[3/8] Selecionando Logistic Regression sem consultar o teste...", flush=True)
    lr_model, lr_info = _fit_lr(x_lr_train, y_train, x_lr_val, y_val)
    print("[4/8] Selecionando Random Forest sem consultar o teste...", flush=True)
    rf_model, rf_info = _fit_rf(x_f32_train, y_train, x_f32_val, y_val)
    print("[5/8] Selecionando MLP Keras sem consultar o teste...", flush=True)
    mlp_model, mlp_config, mlp_info = _fit_mlp_candidates(x_f32_train, y_train, x_f32_val, y_val)

    validation_scores = {
        "Logistic Regression": lr_info["selected_validation"],
        "Random Forest": rf_info["selected_validation"],
        "MLP Keras": mlp_info["selected_validation"],
    }
    champion = max(
        validation_scores,
        key=lambda name: (
            validation_scores[name]["f1_weighted"],
            validation_scores[name]["accuracy"],
        ),
    )
    champion_family = {
        "Logistic Regression": "lr",
        "Random Forest": "rf",
        "MLP Keras": "mlp",
    }[champion]
    selection = {
        "protocol": "selection_on_validation_only",
        "champion": champion,
        "champion_family": champion_family,
        "validation_scores": validation_scores,
        "lr": lr_info,
        "rf": rf_info,
        "mlp": mlp_info,
    }
    _json_dump(root / "artifacts" / "selection.json", selection)

    print("[6/8] Congelamento concluído. Avaliando uma vez no teste independente...", flush=True)
    test_inputs = {
        "Logistic Regression": x_lr_test,
        "Random Forest": x_f32_test,
        "MLP Keras": x_f32_test,
    }
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "MLP Keras": mlp_model,
    }
    families = {"Logistic Regression": "lr", "Random Forest": "rf", "MLP Keras": "mlp"}

    test_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, model in models.items():
        pred, proba, infer_seconds = _predict_with_timing(
            model, test_inputs[name], families[name]
        )
        predictions[name] = pred
        probabilities[name] = proba
        timings[name] = infer_seconds
        test_metrics[name] = classification_metrics(y_test, pred)
        cm = confusion_10x10(y_test, pred)
        safe = name.lower().replace(" ", "_")
        np.savetxt(root / "outputs" / "tables" / f"confusion_{safe}.csv", cm, delimiter=",", fmt="%d")
        top_confusions(cm, top_n=10).to_csv(
            root / "outputs" / "tables" / f"top_confusions_{safe}.csv", index=False
        )
        plot_confusion_matrix(
            cm,
            f"Matriz de confusão — {name}",
            root / "outputs" / "figures" / f"confusion_{safe}.png",
        )

    metrics_table(test_metrics).to_csv(
        root / "outputs" / "tables" / "benchmark_metrics.csv", index=False
    )
    _json_dump(root / "artifacts" / "timings.json", timings)

    print("[7/8] Persistindo e recarregando modelos...", flush=True)
    lr_path = root / "artifacts" / "models" / "logistic_regression.joblib"
    rf_path = root / "artifacts" / "models" / "random_forest.joblib"
    mlp_path = root / "artifacts" / "models" / "mlp.keras"
    joblib.dump(lr_model, lr_path, compress=3)
    joblib.dump(rf_model, rf_path, compress=3)
    mlp_model.save(mlp_path)

    lr_reload = joblib.load(lr_path)
    rf_reload = joblib.load(rf_path)
    mlp_reload = tf.keras.models.load_model(mlp_path)
    reload_checks = {
        "lr": bool(np.array_equal(lr_reload.predict(x_lr_test), predictions["Logistic Regression"])),
        "rf": bool(np.array_equal(rf_reload.predict(x_f32_test), predictions["Random Forest"])),
        "mlp": bool(np.array_equal(
            predict_labels(mlp_reload, x_f32_test), predictions["MLP Keras"]
        )),
    }
    if not all(reload_checks.values()):
        raise AssertionError(f"Reload alterou predições: {reload_checks}")

    print("[8/8] Executando Class Masking/OOD e inferência externa...", flush=True)
    x_mask_train, y_mask_train, known = restrict_training(
        x_f32_train, y_train, DEFAULT_HIDDEN
    )
    x_mask_val, y_mask_val, known_val = restrict_training(
        x_f32_val, y_val, DEFAULT_HIDDEN
    )
    if not np.array_equal(known, known_val):
        raise AssertionError("Mapas de classes inconsistentes no masking.")
    masked_model = build_mlp(output_units=len(known), config=mlp_config, seed=SEED)
    masked_history, masked_fit_seconds = fit_mlp(
        masked_model,
        x_mask_train,
        y_mask_train,
        x_mask_val,
        y_mask_val,
        config=mlp_config,
        epochs=15,
        patience=3,
        verbose=0,
    )
    masked_path = root / "artifacts" / "models" / "mlp_masked_4_7.keras"
    masked_model.save(masked_path)
    np.save(root / "artifacts" / "models" / "masked_known_classes.npy", known)

    hidden_mask = np.isin(y_test, DEFAULT_HIDDEN)
    known_mask = ~hidden_mask
    ood_proba = predict_proba(masked_model, x_f32_test[hidden_mask])
    id_proba = predict_proba(masked_model, x_f32_test[known_mask])
    ood_pred = decode_internal(ood_proba.argmax(axis=1), known)
    id_pred = decode_internal(id_proba.argmax(axis=1), known)
    id_metrics = classification_metrics(y_test[known_mask], id_pred)
    ood_matrix = hidden_vs_known_matrix(
        y_test[hidden_mask], ood_pred, DEFAULT_HIDDEN, known
    )
    plot_confusion_matrix(
        ood_matrix,
        "Classes ocultas 4/7 → classes conhecidas",
        root / "outputs" / "figures" / "ood_hidden_vs_known.png",
        x_labels=known,
        y_labels=np.asarray(DEFAULT_HIDDEN),
    )
    plot_uncertainty_distributions(
        id_proba,
        ood_proba,
        root / "outputs" / "figures" / "ood_msp_id_vs_ood.png",
    )
    high_idx = np.argsort(ood_proba.max(axis=1))[::-1][:12]
    pd.DataFrame(
        {
            "classe_real_oculta": y_test[hidden_mask][high_idx],
            "classe_atribuida": ood_pred[high_idx],
            "confianca_maxima": ood_proba.max(axis=1)[high_idx],
        }
    ).to_csv(root / "outputs" / "tables" / "ood_high_confidence_examples.csv", index=False)

    ood_results = {
        "hidden_classes": list(DEFAULT_HIDDEN),
        "known_classes": known.astype(int).tolist(),
        "masked_model": "MLP Keras 8 outputs",
        "masked_epochs": len(masked_history.history["loss"]),
        "masked_fit_seconds": masked_fit_seconds,
        "id_metrics_known_classes": id_metrics,
        "id_uncertainty": uncertainty_summary(id_proba),
        "ood_uncertainty": uncertainty_summary(ood_proba),
        "ood_samples": int(hidden_mask.sum()),
    }
    _json_dump(root / "artifacts" / "ood_results.json", ood_results)

    own_dir = root / "data" / "own_digits"
    labels_path = own_dir / "labels.json"
    own_result: dict[str, Any] = {"status": "PENDENTE_IMAGEM_PROPRIA"}
    if labels_path.exists():
        config = json.loads(labels_path.read_text(encoding="utf-8"))
        image_path = own_dir / config["file"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        own_result = evaluate_handwritten_sheet(
            root=root,
            image_path=image_path,
            expected_labels=[int(v) for v in config["expected_labels"]],
            threshold=float(config.get("blue_excess_threshold", 10)),
            dilation_iterations=int(config.get("dilation_iterations", 1)),
        )
    elif require_own_image:
        raise FileNotFoundError(
            "Inclua a fotografia manuscrita e labels.json em data/own_digits para concluir o experimento."
        )

    summary = {
        "run_mode": "FULL",
        "seed": SEED,
        "split": {"train": 49000, "validation": 7000, "test": 14000},
        "champion_selected_on_validation": champion,
        "test_metrics": test_metrics,
        "reload_checks": reload_checks,
        "ood": ood_results,
        "own_image": own_result,
        "status": "COMPLETE" if own_result["status"] == "PASS" else "PARTIAL",
    }
    _json_dump(root / "artifacts" / "run_summary.json", summary)
    return summary
