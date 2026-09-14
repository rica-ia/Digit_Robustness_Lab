import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from digit_robustness.ood import decode_internal, hidden_vs_known_matrix, known_classes, predictive_entropy, restrict_training, uncertainty_summary
from digit_robustness.vision import flatten_for_model, process_digit


def test_masking_removes_hidden_classes_and_maps_internal_labels():
    x = np.zeros((10, 4), dtype=np.float32)
    y = np.arange(10, dtype=np.uint8)
    x_known, y_internal, known = restrict_training(x, y, hidden=(4, 7))
    assert x_known.shape[0] == 8
    assert np.array_equal(known, np.array([0, 1, 2, 3, 5, 6, 8, 9]))
    assert np.array_equal(np.unique(y_internal), np.arange(8))
    assert np.array_equal(decode_internal(y_internal, known), known)


def test_uncertainty_and_hidden_matrix():
    p = np.array([[0.8, 0.2], [0.5, 0.5]], dtype=float)
    h = predictive_entropy(p)
    assert h[0] < h[1]
    summary = uncertainty_summary(p)
    assert 0 <= summary["msp_mean"] <= 1
    known = np.array([0, 1, 2, 3, 5, 6, 8, 9])
    cm = hidden_vs_known_matrix(np.array([4, 7]), np.array([3, 9]), (4, 7), known)
    assert cm.shape == (2, 8)
    assert cm.sum() == 2


def test_external_digit_processing_contract():
    image = np.full((90, 70), 255, dtype=np.uint8)
    image[15:75, 28:42] = 0
    processed = process_digit(image)
    assert processed.shape == (28, 28)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1
    cy, cx = center_of_mass(processed)
    assert abs(cy - 13.5) < 1.5 and abs(cx - 13.5) < 1.5
    assert flatten_for_model(processed, "lr").dtype == np.float64
    assert flatten_for_model(processed, "mlp").dtype == np.float32


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        process_digit(np.zeros((28, 28), dtype=np.uint8))
