"""Start the desktop laboratory with ``python -m plenoptic_studio``."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from .gui import launch

raise SystemExit(launch())
