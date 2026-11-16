#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c 'import numpy, scipy, matplotlib, PySide6' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m plenoptic_studio
