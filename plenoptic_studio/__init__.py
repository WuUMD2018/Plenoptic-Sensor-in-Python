"""Plenoptic Studio: laser wavefront simulation and reconstruction."""
import os

# Small, repeated transforms run more predictably without BLAS oversubscription.
# Respect an explicitly configured thread count.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.2.0"
