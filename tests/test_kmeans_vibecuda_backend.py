"""Correctness tests for the ``vibecuda`` k-means backend (Blackwell CUDA).

Same style as ``test_primitives_correctness.py``: every case compares the
fused kernel against an inline fp32 chunked reference that implements the
shared flash-kmeans KDA contract:

* Euclidean assign with lowest-index tie break,
* Lloyd update with previous-centroid keep on empty clusters,
* int32 ids + bf16 centroids.

Skips on non-CUDA or non-Blackwell hosts (the backend's supported regime is
sm100/sm103, bf16, D=128, 2<=K<=1024).
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
blackwell_only = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) < (10, 0),
    reason="vibecuda backend requires sm100/sm103 Blackwell",
)

RTOL_INERTIA = 5e-3   # matches PR23 benchmarks/benchmark_backends.py --rtol


def _ref_assign(x, c):
    """fp32 reference assign with lowest-index tie break."""
    xf, cf = x.float(), c.float()
    cn = (cf * cf).sum(-1)
    pn = (xf * xf).sum(-1, keepdim=True)
    d = pn + cn[:, None, :] - 2.0 * (xf @ cf.transpose(-1, -2))
    return d.min(dim=-1).indices


def _ref_lloyd(x, c, iters):
    """fp32 reference Lloyd with previous-centroid keep on empty clusters."""
    B, N, D = x.shape
    K = c.shape[1]
    cc = c.float()
    ids = None
    for _ in range(iters):
        ids = _ref_assign(x, cc.to(torch.bfloat16))
        sums = torch.zeros(B, K, D, device=x.device, dtype=torch.float32)
        cnts = torch.zeros(B, K, device=x.device, dtype=torch.float32)
        sums.scatter_add_(1, ids[..., None].expand(-1, -1, D), x.float())
        cnts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        cc = torch.where((cnts > 0)[..., None], sums / cnts.clamp_min(1)[..., None], cc)
    return ids, cc


def _inertia(x, ids, cents):
    D = x.shape[-1]
    own = cents.float().gather(1, ids.long()[..., None].expand(-1, -1, D))
    return float(((x.float() - own) ** 2).sum(-1).mean())


@blackwell_only
def test_vibecuda_routes_on_blackwell_bf16_d128():
    """Default dispatch must pick the vibecuda backend in its regime."""
    from flashlib.primitives.kmeans.impl import _route

    backend, _ = _route(B=2, N=2048, D=128, K=64, dtype=torch.bfloat16)
    assert backend == "vibecuda"
    # fp32 stays on the pre-existing default (triton on this host).
    backend_fp32, _ = _route(B=2, N=2048, D=128, K=64, dtype=torch.float32)
    assert backend_fp32 != "vibecuda"


@blackwell_only
def test_vibecuda_explicit_unsupported_shape_fails_loudly():
    """Explicit backend='vibecuda' outside the regime must raise, not reroute."""
    from flashlib import flash_kmeans

    x = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception):
        flash_kmeans(x, 8, backend="vibecuda")  # D=64 unsupported


@blackwell_only
@pytest.mark.parametrize("K", [2, 65, 128, 293, 1024])
def test_vibecuda_assign_matches_reference(K):
    """Fixed-centroid assign: == 100% agreement on ties-free random data."""
    from flashlib import vibecuda_assign_euclid

    torch.manual_seed(0)
    B, N, D = 2, 4096, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(N, device="cuda")[:K] for _ in range(B)]
    )
    c = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()

    ids = vibecuda_assign_euclid(x, c)
    assert ids.dtype == torch.int32 and ids.shape == (B, N)

    ref = _ref_assign(x, c)
    agree = (ids.long() == ref).float().mean().item()
    # fp32 inline scoring vs fused bf16-tile scoring can disagree on a
    # handful of exact ties; the shared contract is the >0.99 gate.
    assert agree >= 0.9995, f"assign agreement {agree:.6f} at K={K}"
    torch.cuda.synchronize()
    # Input preservation: kernel must not mutate x or centroids.
    x2 = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    x_copy = x2.clone()
    c_copy = c.clone()
    vibecuda_assign_euclid(x2, c)
    assert torch.equal(x2, x_copy) and torch.equal(c, c_copy)


@blackwell_only
@pytest.mark.parametrize("iters", [1, 2])
def test_vibecuda_lloyd_inertia_matches_reference(iters):
    """Lloyd: inertia relative difference vs fp32 reference within PR23 rtol."""
    from flashlib import flash_kmeans

    torch.manual_seed(1)
    B, N, D, K = 4, 8192, 128, 256
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, N, (B, K), device="cuda")
    init = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()

    ids, cents, n_iter = flash_kmeans(
        x, K, max_iters=iters, tol=0.0, init_centroids=init, backend="vibecuda"
    )
    assert n_iter == iters
    assert ids.dtype == torch.int32 and cents.dtype == torch.bfloat16

    ref_ids, ref_c = _ref_lloyd(x, init, iters)
    cand_inertia = _inertia(x, ids, cents)
    ref_inertia = _inertia(x, ref_ids, ref_c)
    rel = abs(cand_inertia - ref_inertia) / ref_inertia
    assert rel < RTOL_INERTIA, f"inertia rel diff {rel:.3e} >= {RTOL_INERTIA}"
    if iters == 1:
        agree = (ids.long() == ref_ids).float().mean().item()
        assert agree > 0.99, f"one-iteration agreement {agree:.4f}"


_POISON_INT = 1 << 30  # outside any valid label range [0, K)


@blackwell_only
@pytest.mark.parametrize(
    "B,N,K", [(2, 257, 3), (2, 2048, 96), (4, 8192, 256)],
    ids=["plain-assign", "plain-mid", "umma-assign"],
)
@pytest.mark.parametrize("iters", [1, 2, 3])
def test_vibecuda_lloyd_fully_writes_caller_output_buffers(B, N, K, iters):
    """Deterministic full-write proof via caller-owned sentinel buffers.

    Caller allocates the exact output tensors, poisons every element with a
    sentinel, and hands them to the test-only ``_kmeans_lloyd_into`` binding,
    which makes the kernels write the final assign labels and the final
    centroid update DIRECTLY into those buffers (no allocator heuristic, no
    post-hoc copy). Any unwritten element keeps its sentinel, so a stale,
    empty, or partially written output cannot pass — for BOTH outputs, on
    both assign-kernel branches (N<4096 plain and N>=4096 UMMA), and for
    odd/even iteration counts (final-update buffer parity).
    """
    from flashlib import flash_kmeans
    from flashlib.primitives.kmeans.vibecuda.impl import _kmeans_lloyd_into

    torch.manual_seed(2)
    D = 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, N, (B, K), device="cuda")
    init = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()
    before_x = x.clone()

    out_ids = torch.full((B, N), _POISON_INT, device="cuda", dtype=torch.int32)
    out_cents = torch.full((B, K, D), float("nan"), device="cuda",
                           dtype=torch.bfloat16)
    _kmeans_lloyd_into(x, init, iters, out_ids, out_cents)
    torch.cuda.synchronize()

    n_surv_ids = int((out_ids.long() == _POISON_INT).sum())
    assert n_surv_ids == 0, \
        f"int sentinel survived in {n_surv_ids} ids elements: not fully written"
    n_surv_c = int(torch.isnan(out_cents.float()).sum())
    assert n_surv_c == 0, \
        f"NaN sentinel survived in {n_surv_c} centroid elements: not fully written"
    assert (out_ids >= 0).all() and (out_ids < K).all(), "ids out of range"
    assert torch.equal(x, before_x), "input x mutated"

    # The deterministic-pipeline test binding must reproduce the production
    # public-API result bit-for-bit (same kernels, same schedule).
    ids_pub, cents_pub, _ = flash_kmeans(
        x, K, max_iters=iters, tol=0.0, init_centroids=init, backend="vibecuda"
    )
    assert torch.equal(out_ids, ids_pub), "into-binding labels != public API"
    assert torch.equal(out_cents, cents_pub), "into-binding centroids != public API"


@blackwell_only
@pytest.mark.parametrize(
    "B,N,K", [(2, 257, 3), (2, 2048, 96), (4, 8192, 256)],
    ids=["plain-assign", "plain-mid", "umma-assign"],
)
def test_vibecuda_predict_fully_writes_caller_label_buffer(B, N, K):
    """Deterministic full-write proof for fixed-centroid predict.

    Predict's only produced output is the int32 label tensor; the centroids
    it returns are the unchanged input tensor, which is NOT treated as a
    produced output and is not part of this proof.
    """
    from flashlib import vibecuda_assign_euclid
    from flashlib.primitives.kmeans.vibecuda.impl import _kmeans_lloyd_into

    torch.manual_seed(3)
    D = 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, N, (B, K), device="cuda")
    init = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()

    out_ids = torch.full((B, N), _POISON_INT, device="cuda", dtype=torch.int32)
    _kmeans_lloyd_into(x, init, 0, out_ids, None)
    torch.cuda.synchronize()

    n_surv = int((out_ids.long() == _POISON_INT).sum())
    assert n_surv == 0, \
        f"int sentinel survived in {n_surv} predict ids: not fully written"
    assert (out_ids >= 0).all() and (out_ids < K).all(), "ids out of range"

    pred_pub = vibecuda_assign_euclid(x, init)
    assert torch.equal(out_ids, pred_pub), "into-binding labels != public API"


@blackwell_only
def test_vibecuda_do_not_move_boundaries():
    """Boundary contract rows from the flash-kmeans PR23 test matrix."""
    from flashlib import flash_kmeans, vibecuda_assign_euclid

    for B, N, K in [(2, 2, 2), (2, 257, 3), (2, 130, 128), (2, 1024, 591), (2, 1024, 1024)]:
        torch.manual_seed(3)
        x = torch.randn(B, N, 128, device="cuda", dtype=torch.bfloat16)
        idx = torch.stack(
            [torch.randperm(N, device="cuda")[:K] for _ in range(B)]
        )
        init = torch.gather(x, 1, idx[..., None].expand(-1, -1, 128)).contiguous()
        ids, cents, n_iter = flash_kmeans(
            x, K, max_iters=1, tol=0.0, init_centroids=init, backend="vibecuda"
        )
        ref_ids, ref_c = _ref_lloyd(x, init, 1)
        agree = (ids.long() == ref_ids).float().mean().item()
        ref_inertia = _inertia(x, ref_ids, ref_c)
        if ref_inertia > 0:
            rel = abs(_inertia(x, ids, cents) - ref_inertia) / ref_inertia
        else:
            # Every point is its own centroid: both inertias must be zero.
            rel = _inertia(x, ids, cents)
        assert agree > 0.99, f"B{B} N{N} K{K}: agreement {agree:.4f}"
        assert rel < RTOL_INERTIA, f"B{B} N{N} K{K}: inertia rel {rel:.3e}"

        p = vibecuda_assign_euclid(x, init)
        ref_p = _ref_assign(x, init)
        assert (p.long() == ref_p).float().mean().item() >= 0.9995
