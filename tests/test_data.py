import numpy as np

from digit_robustness.data import make_split_indices, normalize_for_family


def test_split_has_expected_sizes_and_no_overlap():
    y = np.tile(np.arange(10, dtype=np.uint8), 7000)
    splits = make_split_indices(y, random_state=42)
    assert len(splits["train"]) == 49000
    assert len(splits["validation"]) == 7000
    assert len(splits["test"]) == 14000
    assert np.intersect1d(splits["train"], splits["validation"]).size == 0
    assert np.intersect1d(splits["train"], splits["test"]).size == 0
    assert np.intersect1d(splits["validation"], splits["test"]).size == 0


def test_normalization_dtype_by_family():
    raw = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    lr = normalize_for_family(raw, "lr")
    rf = normalize_for_family(raw, "rf")
    mlp = normalize_for_family(raw, "mlp")
    assert lr.dtype == np.float64
    assert rf.dtype == np.float32
    assert mlp.dtype == np.float32
    assert lr.min() == 0 and lr.max() == 1
    assert rf.min() == 0 and rf.max() == 1
