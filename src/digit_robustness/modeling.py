"""Construção e treinamento das três famílias de modelos."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


@dataclass(frozen=True)
class MLPConfig:
    hidden1: int = 128
    hidden2: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 128


def build_logistic_regression(C: float = 1.0, solver: str = "lbfgs") -> LogisticRegression:
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=400,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )


def build_random_forest(
    n_estimators: int = 160,
    max_depth: int | None = None,
    n_jobs: int = 2,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        class_weight=None,
    )


def build_mlp(
    output_units: int = 10,
    config: MLPConfig = MLPConfig(),
    seed: int = RANDOM_STATE,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(config.hidden1, activation="relu"),
            tf.keras.layers.Dense(config.hidden2, activation="relu"),
            tf.keras.layers.Dense(output_units, activation="softmax"),
        ],
        name="digit_mlp",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_mlp(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: MLPConfig,
    epochs: int = 15,
    patience: int = 3,
    verbose: int = 0,
) -> tuple[tf.keras.callbacks.History, float]:
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.batch_size,
        callbacks=[callback],
        verbose=verbose,
        shuffle=True,
    )
    return history, time.perf_counter() - start


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    """Uniformiza predict_proba entre sklearn e Keras."""
    if isinstance(model, tf.keras.Model):
        proba = model(x, training=False).numpy()
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
    else:
        raise TypeError("Modelo não fornece probabilidades.")
    proba = np.asarray(proba)
    if proba.ndim != 2 or not np.isfinite(proba).all():
        raise ValueError("Probabilidades inválidas.")
    return proba


def predict_labels(model: Any, x: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    if isinstance(model, tf.keras.Model):
        idx = predict_proba(model, x).argmax(axis=1)
        return idx if classes is None else np.asarray(classes)[idx]
    return np.asarray(model.predict(x))
