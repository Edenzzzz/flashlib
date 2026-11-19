"""kmeans vibecuda backend (native CUDA C++, SM100/SM103).

Re-exports the public Python wrappers from ``impl.py``. The CUDA sources
under ``csrc/`` are JIT-compiled once at first use via
``torch.utils.cpp_extension.load`` (ninja-cached); :func:`precompile` moves
that one-time build out of the first timed call when desired.
"""
from flashlib.primitives.kmeans.vibecuda.impl import (
    VibeCudaKmeansUnsupported,
    precompile,
    vibecuda_assign_euclid,
    vibecuda_kmeans_Euclid,
    vibecuda_kmeans_support_error,
    vibecuda_kmeans_supported,
)

__all__ = [
    "VibeCudaKmeansUnsupported",
    "precompile",
    "vibecuda_kmeans_Euclid",
    "vibecuda_assign_euclid",
    "vibecuda_kmeans_supported",
    "vibecuda_kmeans_support_error",
]
