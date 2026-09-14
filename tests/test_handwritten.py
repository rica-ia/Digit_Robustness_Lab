import json
from pathlib import Path

import numpy as np

from digit_robustness.handwritten import load_rgb, prepare_component, segment_blue_digits

ROOT = Path(__file__).resolve().parents[1]


def test_blue_ballpoint_sheet_segmentation_and_preprocessing():
    config = json.loads((ROOT / "data" / "own_digits" / "labels.json").read_text(encoding="utf-8"))
    image = load_rgb(ROOT / "data" / "own_digits" / config["file"])
    mask, components = segment_blue_digits(image, threshold=config["blue_excess_threshold"])
    assert len(components) == len(config["expected_labels"]) == 14
    processed = prepare_component(mask, components[0], dilation_iterations=config["dilation_iterations"])
    assert processed.shape == (28, 28)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min()) <= float(processed.max()) <= 1.0
