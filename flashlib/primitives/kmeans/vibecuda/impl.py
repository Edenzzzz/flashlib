"""VibeCUDA fused K-means backend (SM100/SM103, native CUDA C++).

A single custom CUDA C++ extension implements the full public KDA Lloyd
contract from flash-kmeans PR 23 on Blackwell (sm100/sm103):

* iterative BF16 ``D=128`` Lloyd clustering (assign + centroid update) for
  ``B >= 1``, ``N >= 2`` and ``2 <= K <= 1024``, and
* fixed-centroid assignment ("predict") under the same shape rules.

Both entry points accept explicit initial centroids and return int32 cluster
ids + bf16 centroids, matching the reference semantics of
``batch_kmeans_Euclid``/``euclid_assign`` in flash-kmeans: first-min-index
tie breaking (lowest index wins), per-batch centering of scores, and an
empty-cluster policy that keeps the previous centroid.

Per-architecture CUDA sources live under ``csrc/`` and are JIT-compiled once
at first use (ninja-cached); the timed call path only enqueues prebuilt
kernels with no host synchronization. Dependency: PyTorch's C++ extension
loader only (``torch.utils.cpp_extension.load``), same JIT convention as the
CuteDSL backend.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from flashlib import _hw

_SUPPORTED_MIN_MAJOR = 10
_SUPPORTED_MIN_CAP = (10, 0)

_CSRC = Path(__file__).resolve().parent / "csrc"


class VibeCudaKmeansUnsupported(RuntimeError):
    """Raised when the ``vibecuda`` backend is asked to run outside its regime."""


def _ensure_extension_loader_importable() -> None:
    """Make ``torch.utils.cpp_extension`` importable in hostile environments.

    Importing it triggers ``setuptools`` -> ``pkg_resources``, whose startup
    scan crashes with ``PermissionError`` when an inherited ``sys.path`` entry
    contains an unreadable ``*.egg-info``/``*.dist-info`` directory (observed
    in some sandboxed CI/runner host images). When that happens, temporarily
    remove only the offending entries, import ``pkg_resources`` once (its
    startup scan runs at import time), then restore ``sys.path`` — the now
    cached module makes every later ``cpp_extension`` import in this process
    succeed. When no such entry exists this is a no-op passthrough.
    """
    try:
        import torch.utils.cpp_extension  # noqa: F401
        return
    except PermissionError:
        pass
    removed = []
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            children = os.listdir(entry)
        except OSError:
            continue  # unreadable top entries do not reach the failing scan
        for child in children:
            if not child.endswith((".egg-info", ".dist-info")):
                continue
            try:
                os.listdir(os.path.join(entry, child))
            except OSError:
                sys.path.remove(entry)
                removed.append(entry)
                break
    try:
        import pkg_resources  # noqa: F401
    finally:
        sys.path[0:0] = removed
    import torch.utils.cpp_extension  # noqa: F401


def vibecuda_kmeans_support_error(
    x: torch.Tensor, K: int, hw: Optional[_hw.HwProps] = None
) -> Optional[str]:
    """Return ``None`` when the vibecuda kernel can serve this call, else a
    human-readable reason. Used by the dispatcher and by
    :func:`vibecuda_kmeans_supported`.
    """
    hw = hw or _hw.current()
    cap = (hw.sm_arch // 10, hw.sm_arch % 10)
    if cap < _SUPPORTED_MIN_CAP:
        return (
            f"vibecuda backend requires SM10x (sm100/sm103 Blackwell), got "
            f"{hw.name!r} (sm_arch={hw.sm_arch})"
        )
    if x.ndim == 2:
        D = x.shape[-1]
    elif x.ndim == 3:
        D = x.shape[-1]
    else:
        return f"vibecuda backend expects (B, N, D) or (N, D) input, got ndim={x.ndim}"
    if x.dtype != torch.bfloat16:
        return f"vibecuda backend requires bf16 input, got {x.dtype}"
    if D != 128:
        return f"vibecuda backend requires D=128, got D={D}"
    if not (2 <= K <= 1024):
        return f"vibecuda backend requires 2 <= K <= 1024, got K={K}"
    return None


def vibecuda_kmeans_supported(x: torch.Tensor, K: int) -> bool:
    """True when :func:`vibecuda_kmeans_Euclid` can run this shape/dtype."""
    return vibecuda_kmeans_support_error(x, K) is None


@lru_cache(maxsize=2)
def _load_extension(sm_arch: int):
    """JIT-compile the per-architecture CUDA source exactly once.

    Independent per-architecture sources are kept deliberately: tuning that
    helped one Blackwell variant sometimes regressed the other, so each ships
    its own retained implementation selected by device capability.
    """
    cap = (sm_arch // 10, sm_arch % 10)
    if cap < _SUPPORTED_MIN_CAP:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires SM10x (sm100/sm103 Blackwell), got sm_{sm_arch}"
        )
    arch_tag = f"{cap[0]}{cap[1]}a"  # (10,0)->100a, (10,3)->103a
    src_file = "kmeans_lloyd_sm103.cu" if cap >= (10, 3) else "kmeans_lloyd_sm100.cu"

    # Build only for the architecture of the current device: compiling a
    # second gencode doubles extension compile time. This mirrors the
    # repository's other JIT backends which compile for the current device
    # on first use.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}a")

    _ensure_extension_loader_importable()
    from torch.utils.cpp_extension import load as _load_ext

    return _load_ext(
        name=f"flashlib_vibecuda_kmeans_sm{arch_tag}",
        sources=[str(_CSRC / src_file)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            f"-gencode=arch=compute_{arch_tag},code=sm_{arch_tag}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _extension():
    hw = _hw.current()
    if not hw.is_cuda:
        raise VibeCudaKmeansUnsupported(
            "vibecuda backend requires a CUDA device (dispatching on CPU)"
        )
    return _load_extension(hw.sm_arch)


def precompile() -> None:
    """Eagerly JIT-build the vibecuda k-means extension for the current device.

    Optional warmup helper: the public entry points already build lazily on
    first use (ninja-cached), so calling this only moves the one-time compile
    earlier, e.g. into application startup. This is the only build entry
    point callers need; it takes no arguments and selects the per-architecture
    CUDA source from the current device's compute capability, exactly as the
    lazy path does. A repeated call is a cache hit and returns immediately.
    """
    _extension()


@lru_cache(maxsize=2)
def _load_probe_extension(sm_arch: int):
    """JIT-compile the TEST-ONLY probe extension (with ``kda_lloyd_into``).

    Kept deliberately separate from the production extension: the sentinel
    full-write binding gets its own translation unit and shared object so it
    cannot perturb production code generation or loading. The probe and
    production extensions include the same shared implementation header, so
    the pipeline they execute is line-identical.
    """
    cap = (sm_arch // 10, sm_arch % 10)
    if cap < _SUPPORTED_MIN_CAP:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires SM10x (sm100/sm103 Blackwell), got sm_{sm_arch}"
        )
    arch_tag = f"{cap[0]}{cap[1]}a"  # (10,0)->100a, (10,3)->103a
    probe_src = "kmeans_lloyd_probe_sm103.cu" if cap >= (10, 3) else "kmeans_lloyd_probe_sm100.cu"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}a")

    _ensure_extension_loader_importable()
    from torch.utils.cpp_extension import load as _load_ext

    return _load_ext(
        name=f"flashlib_vibecuda_kmeans_probe_sm{arch_tag}",
        sources=[str(_CSRC / probe_src)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            f"-gencode=arch=compute_{arch_tag},code=sm_{arch_tag}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _probe_extension():
    hw = _hw.current()
    if not hw.is_cuda:
        raise VibeCudaKmeansUnsupported(
            "vibecuda backend requires a CUDA device (dispatching on CPU)"
        )
    return _load_probe_extension(hw.sm_arch)


def vibecuda_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
):
    """Batched Euclidean Lloyd clustering via the fused CUDA kernel.

    Identical interface to ``flash_kmeans.batch_kmeans_Euclid``:
    returns ``(cluster_ids int32 (B, N), centroids bf16 (B, K, D), n_iter)``.

    The kernel always runs exactly ``max_iters`` assign/update rounds, so
    convergence-based early exit is not available: ``tol`` must be ``0.0``
    (match this by pre-checking in the dispatcher).
    """
    if x.ndim != 3:
        raise ValueError(
            f"vibecuda backend operates on (B, N, D) batches, got ndim={x.ndim}; "
            "unsqueeze via flashlib.flash_kmeans for (N, D) input"
        )
    # Every validation clause from the canonical ``vibecuda_kmeans_support_error``
    # is applied inline on every call (identical semantics, no call-frame or
    # re-probe cost); the hardware fingerprint is fetched once and reused for
    # both the support check and the per-architecture extension lookup so the
    # timed fast path never pays for redundant device queries.
    sm_arch = _hw.current().sm_arch
    if sm_arch < 100:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires SM10x (sm100/sm103 Blackwell), got "
            f"sm_arch={sm_arch}"
        )
    if x.dtype != torch.bfloat16:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires bf16 input, got {x.dtype}"
        )
    if x.shape[-1] != 128:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires D=128, got D={x.shape[-1]}"
        )
    if not (2 <= n_clusters <= 1024):
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires 2 <= K <= 1024, got K={n_clusters}"
        )
    if tol != 0.0:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend runs a fixed iteration count (tol must be 0.0, got {tol})"
        )
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1 for Lloyd, got {max_iters}")

    B, N, D = x.shape
    K = n_clusters
    if N < 2:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires N >= 2, got N={N}"
        )

    if init_centroids is None:
        indices = torch.randint(0, N, (B, K), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=indices[..., None].expand(-1, -1, D)
        ).contiguous()
    else:
        centroids = init_centroids
        if centroids.dtype != torch.bfloat16:
            raise VibeCudaKmeansUnsupported(
                f"vibecuda backend requires bf16 initial centroids, got {centroids.dtype}"
            )

    # ``sm_arch >= 100`` here (the arch check raised otherwise, which also
    # covers the CPU-dispatch case: the CPU sentinel has sm_arch=0).
    ext = _load_extension(sm_arch)
    ids, cur = ext.kda_lloyd(x.contiguous(), centroids.contiguous(), int(max_iters))
    iters = int(max_iters)
    if verbose:
        # The assign/update loop is fused GPU-side in one extension call, so
        # per-iteration shift logging is not available; the fused kernel
        # always runs the exact requested count.
        print(f"vibecuda backend: ran {iters} fused Lloyd iterations")
    return ids, cur, iters


def vibecuda_assign_euclid(
    x: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Fixed-centroid assignment: nearest-centroid id per row (int32).

    Mirrors ``euclid_assign`` from flash-kmeans under the same contract
    (bf16, D=128, B>=1, N>=2, 2<=K<=1024, lowest-index tie break).
    """
    if x.ndim == 2:
        x_b = x.unsqueeze(0)
        c_b = centroids.unsqueeze(0) if centroids.ndim == 2 else centroids
        squeeze = True
    elif x.ndim == 3:
        x_b = x
        c_b = centroids.unsqueeze(0) if centroids.ndim == 2 else centroids
        squeeze = False
    else:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend expects (B, N, D) or (N, D) input, got ndim={x.ndim}"
        )
    sm_arch = _hw.current().sm_arch
    if sm_arch < 100:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires SM10x (sm100/sm103 Blackwell), got "
            f"sm_arch={sm_arch}"
        )
    if x_b.dtype != torch.bfloat16:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires bf16 input, got {x_b.dtype}"
        )
    if x_b.shape[-1] != 128:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires D=128, got D={x_b.shape[-1]}"
        )
    if not (2 <= c_b.shape[1] <= 1024):
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires 2 <= K <= 1024, got K={c_b.shape[1]}"
        )
    if centroids.dtype != torch.bfloat16:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires bf16 centroids, got {centroids.dtype}"
        )
    if x_b.shape[1] < 2:
        raise VibeCudaKmeansUnsupported(
            f"vibecuda backend requires N >= 2, got N={x_b.shape[1]}"
        )
    ext = _load_extension(sm_arch)
    ids, _ = ext.kda_lloyd(x_b.contiguous(), c_b.contiguous(), 0)
    return ids.squeeze(0) if squeeze else ids


def _kmeans_lloyd_into(
    x: torch.Tensor,
    centroids: torch.Tensor,
    iterations: int,
    out_ids: torch.Tensor,
    out_cents: Optional[torch.Tensor] = None,
) -> None:
    """TEST-ONLY: run the fused pipeline writing into caller-owned buffers.

    Identical computation to :func:`vibecuda_kmeans_Euclid`, but the kernel
    writes the final assign labels into ``out_ids`` and (for Lloyd mode) the
    final centroid update directly into ``out_cents`` — no post-hoc copy, so
    pre-filling these buffers with sentinel values deterministically proves
    full-element writes. This binding exists so the repository tests and
    benchmark can run a poison-then-overwrite proof without relying on the
    caching allocator's block-reuse behavior.

    In predict mode (``iterations == 0``) centroids are not a produced
    output; ``out_cents`` is ignored and must be ``None``.
    """
    reason = vibecuda_kmeans_support_error(x, centroids.shape[1])
    if reason is not None:
        raise VibeCudaKmeansUnsupported(reason)
    if iterations <= 0:
        if out_cents is not None:
            raise ValueError(
                "predict mode does not produce centroids; out_cents must be None"
            )
        dummy = torch.empty(0, dtype=torch.bfloat16)
        _probe_extension().kda_lloyd_into(x.contiguous(), centroids.contiguous(), 0,
                                          out_ids, dummy)
        return
    if out_cents is None:
        raise ValueError("Lloyd mode produces centroids; out_cents is required")
    _probe_extension().kda_lloyd_into(x.contiguous(), centroids.contiguous(),
                                      int(iterations), out_ids, out_cents)


__all__ = [
    "VibeCudaKmeansUnsupported",
    "precompile",
    "vibecuda_kmeans_Euclid",
    "vibecuda_assign_euclid",
    "vibecuda_kmeans_supported",
    "vibecuda_kmeans_support_error",
]
