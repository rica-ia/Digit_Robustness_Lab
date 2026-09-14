#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -n "${DIGIT_PYTHON:-}" ]]; then
  portal_python="$DIGIT_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  portal_python=".venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  portal_python="$VIRTUAL_ENV/bin/python"
else
  echo "Ambiente não encontrado. Na raiz do projeto, execute:"
  echo "python3.12 -m venv .venv"
  echo "source .venv/bin/activate"
  echo "python -m pip install -r requirements.txt"
  echo "python -m pip install -e ."
  echo "Depois execute: bash iniciar_portal.sh"
  exit 1
fi
if [[ ! -x "$portal_python" ]]; then
  echo "Interpretador não encontrado ou não executável: $portal_python"
  exit 1
fi
echo "Portal: http://127.0.0.1:7860"
echo "Gradio original: http://127.0.0.1:7860/gradio"
echo "Encerre com Ctrl+C."
exec "$portal_python" app/app.py "$@"
