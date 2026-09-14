from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from digit_robustness.pipeline import run_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o experimento FULL do Digit Robustness Lab.")
    parser.add_argument("--require-own-image", action="store_true")
    args = parser.parse_args()
    summary = run_full(ROOT, require_own_image=args.require_own_image)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
