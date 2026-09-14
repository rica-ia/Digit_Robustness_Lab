from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from digit_robustness.handwritten import evaluate_handwritten_sheet


def main() -> None:
    config_path = ROOT / "data" / "own_digits" / "labels.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = evaluate_handwritten_sheet(
        ROOT,
        ROOT / "data" / "own_digits" / config["file"],
        [int(v) for v in config["expected_labels"]],
        threshold=float(config.get("blue_excess_threshold", 10)),
        dilation_iterations=int(config.get("dilation_iterations", 1)),
    )
    summary_path = ROOT / "artifacts" / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["own_image"] = result
        # Reavaliar a foto não comprova conclusão das outras etapas.
        # Preserve o status científico registrado pelo pipeline FULL.
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
