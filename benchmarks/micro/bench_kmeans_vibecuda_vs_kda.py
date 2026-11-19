"""Repository-local SOTA comparison: vibecuda vs flash-kmeans PR 23 (cute).

Benchmarks the integrated FlashLib ``vibecuda`` backend against the task's
configured fast baseline — the pinned flash-kmeans PR 23 CuTe backend
(``flash_kmeans.batch_kmeans_Euclid`` / assign path, provenance
svg-project/flash-kmeans#23@97ed5131). Two suites are kept deliberately
separate:

* ``pr23`` — an exact reproduction of the authoritative upstream benchmark
  ``benchmarks/benchmark_backends.py`` from the pinned PR 23 checkout:
  ``torch.manual_seed(0)``; input generated directly as BF16 Gaussian via
  ``torch.randn(..., dtype=torch.bfloat16)`` (no fp32 intermediate);
  initial centroids gathered from ``torch.randint`` row indices; default
  geometry B=4, N=8192, K=64; cold regime (10 Lloyd iterations from the
  sampled initial centroids) and warm regime (2 iterations restarting from
  the cold solution's centroids, matching the production frame-to-frame
  reuse path); per-repeat CUDA-event timing with a host ``synchronize``
  after every repeat, warmup=2, repeats=5, median; cross-backend inertia
  gate at rtol=5e-3 plus label agreement, exactly as upstream compares its
  two backends. The speedup denominator is the configured PR 23 ``cute``
  baseline; PR 23's own ``triton`` backend is run as a diagnostic reference
  leg mirroring the upstream script's comparison pair.

* ``fixed23`` — the 23-row workload portfolio covering LongCat-Video,
  Wan2.1-720p, and HunyuanVideo-480p/720p sparse-attention clustering shapes,
  do-not-move boundary shapes, and fixed-centroid predict shapes, with the
  exact input generators (fp32 Gaussian -> bf16, per-row seeds, randperm-vs-
  randint initial-centroid rule per row class), dtypes, layouts, Lloyd and
  predict modes, iteration counts, and precision checks (fp32-reference
  label agreement for 1-iter/predict rows, relative inertia for
  multi-iteration rows). Timing uses FlashInfer CUPTI, cold L2, 5 warmups,
  10 samples, median. Cache flushing occurs outside each measured activity
  range.

Both legs in every suite time the entire public operator call including
every requested Lloyd iteration. Allocation, planning, JIT compilation,
and input construction happen outside timing; both legs pay identical
in-call allocation costs through their public APIs, so the timed boundary
is symmetric, matching how PR 23's own script times its backends.

Correctness: before any timing, each leg passes an untimed full-write
proof. The vibecuda candidate is proven DETERMINISTICALLY: caller-owned
output buffers prefilled with sentinels (NaN bf16 centroids, 2**30 int32
labels) are passed to a test-only ``kda_lloyd_into`` extension binding that
makes the final assign/update kernels write directly into those buffers, so
every element of every produced output must have been kernel-written (no
sentinel may survive). The pinned baseline legs (immutable sources) use the
allocator-pool variant: blocks of the exact output footprint are poisoned
and freed so the operator's fresh allocations reuse them; every returned
element must have been overwritten. Predict mode proves only the label
tensor — the returned centroids are the unchanged input, not a produced
output. Outputs are additionally validated independently against an
exhaustive fp32 nearest-centroid reference.

Run from the repository checkout root:

    PYTHONPATH=<repo>:<quack>:<flash_kmeans_pr23> \
        python benchmarks/micro/bench_kmeans_vibecuda_vs_kda.py \
        --suite both --out results_kmeans.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

# The pinned PR 23 baseline imports through ``flash_kmeans``; the shared
# venv does not ship its quack dependency, so allow the caller to point at a
# source checkout through QUACK_REPO / FLASH_KMEANS_REPO without installing
# anything. The import is LAZY so a candidate-only run (``--omit-baseline``,
# or an environment where the pinned checkout is intentionally unavailable)
# still exercises every vibecuda leg: correctness gates, full-write proofs,
# and CUPTI timing. A developer/CI machine with the PR 23 checkout on the
# path runs the complete comparison by simply omitting the flag.
_baseline_module = None
batch_kmeans_Euclid = None
euclid_assign = None


def load_baseline():
    """Import the pinned PR 23 baseline; returns True when available."""
    global _baseline_module, batch_kmeans_Euclid, euclid_assign
    if _baseline_module is not None:
        return True
    for _env in ("QUACK_REPO", "FLASH_KMEANS_REPO"):
        _p = os.environ.get(_env, "")
        if _p and Path(_p).is_dir() and _p not in sys.path:
            sys.path.insert(0, _p)
    import flash_kmeans  # noqa: E402  pinned PR 23 baseline
    from flash_kmeans import batch_kmeans_Euclid as _bke  # noqa: E402
    from flash_kmeans.kmeans_api import euclid_assign as _ea  # noqa: E402
    _baseline_module = flash_kmeans
    batch_kmeans_Euclid = _bke
    euclid_assign = _ea
    return True


import flashlib  # noqa: E402  repository checkout (cwd / PYTHONPATH)


# ---------------------------------------------------------------------------
# Genuine prefill / full-element write proof
# ---------------------------------------------------------------------------

_POISON_INT = 1 << 30  # outside any valid label range [0, K)


def prove_full_write(fn, out_meta):
    """Poison-then-overwrite proof that ``fn()`` fully writes every output.

    For each expected output (shape, dtype, kind), allocate several tensors
    of exactly that footprint through the caching allocator, fill them with
    a sentinel (NaN for floats, ``_POISON_INT`` for ints), and free them so
    the operator's fresh output allocations reuse those poisoned blocks.
    Then run ``fn()`` once and assert NO sentinel survives in any returned
    tensor — i.e. every element was genuinely written by the call. A stale,
    empty, or partially written output would retain the poison.

    Allocator-reuse heuristic: used ONLY for the pinned baseline legs, whose
    C++/CuTe sources are immutable and cannot be instrumented. The vibecuda
    candidate uses the deterministic caller-buffer proof in
    :func:`prove_full_write_vibecuda` instead.

    Returns the outputs of ``fn()``.
    """
    poisoned = []
    for shape, dtype, kind in out_meta:
        for _ in range(4):  # several blocks per size class for robust reuse
            t = torch.empty(shape, device="cuda", dtype=dtype)
            if kind == "float":
                t.fill_(float("nan"))
            else:
                t.fill_(_POISON_INT)
            poisoned.append(t)
    del poisoned
    outs = fn()
    flat = [t for t in outs if isinstance(t, torch.Tensor)]
    assert len(flat) >= len(out_meta), "fewer tensor outputs than expected"
    for (shape, dtype, kind), t in zip(out_meta, flat):
        assert isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == dtype
        if kind == "float":
            assert not torch.isnan(t.float()).any(), \
                "NaN sentinel survived: centroid output not fully written"
        else:
            assert not (t.long() == _POISON_INT).any(), \
                "int sentinel survived: label output not fully written"
    return outs


def prove_full_write_vibecuda(x, init, mode, K, iters):
    """Deterministic full-write proof for the vibecuda candidate.

    The caller allocates the exact output tensors, poisons every element
    with a sentinel (NaN bf16 centroids, 2**30 int32 labels), and the
    test-only ``_kmeans_lloyd_into`` binding makes the kernels write the
    final assign labels and final centroid update DIRECTLY into those
    caller-owned buffers — no allocator reuse involved. No sentinel may
    survive. Predict mode's only produced output is the label tensor (the
    returned centroids are the unchanged INPUT, not a produced output), so
    only labels are proven there. Returns the produced outputs.
    """
    from flashlib.primitives.kmeans.vibecuda.impl import _kmeans_lloyd_into

    B, N, D = x.shape
    out_ids = torch.full((B, N), _POISON_INT, device=x.device, dtype=torch.int32)
    if mode == "predict":
        _kmeans_lloyd_into(x, init, 0, out_ids, None)
        torch.cuda.synchronize()
        assert not (out_ids.long() == _POISON_INT).any(), \
            "int sentinel survived: vibecuda predict labels not fully written"
        return (out_ids, init)
    out_cents = torch.full((B, K, D), float("nan"), device=x.device,
                           dtype=torch.bfloat16)
    _kmeans_lloyd_into(x, init, iters, out_ids, out_cents)
    torch.cuda.synchronize()
    assert not (out_ids.long() == _POISON_INT).any(), \
        "int sentinel survived: vibecuda lloyd labels not fully written"
    assert not torch.isnan(out_cents.float()).any(), \
        "NaN sentinel survived: vibecuda lloyd centroids not fully written"
    return (out_ids, out_cents)


# ---------------------------------------------------------------------------
# Shared reference / correctness helpers
# ---------------------------------------------------------------------------

def ref_assign_fp32(x, centroids):
    """Exhaustive fp32 nearest-centroid labels."""
    d = torch.cdist(x.float(), centroids.float())  # (B, N, K)
    return d.argmin(dim=-1).to(torch.int32)


def label_agreement(a, b):
    return (a == b).float().mean().item()


def rel_inertia(x, centroids, ids):
    """Sum of squared distances of assigned centroids, all in fp32."""
    c = centroids.float()
    gathered = torch.gather(
        c, 1, ids.long()[..., None].expand(-1, -1, c.shape[-1]))
    return ((x.float() - gathered) ** 2).sum().item()


# ===========================================================================
# Suite 1: exact PR 23 benchmarks/benchmark_backends.py reproduction
# ===========================================================================

def pr23_cuda_time(fn, warmup, repeats, device):
    """Byte-for-byte the upstream PR 23 timing helper: per-repeat event pair
    with a host synchronize after every repeat, median of samples."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run_pr23_suite(device, warmup=2, repeats=5, rtol=5e-3, baseline_ok=True):
    """Reproduce PR 23 benchmark_backends.py exactly (synthetic-data path).

    Same seed(0), same direct BF16 Gaussian input, same randint initial
    centroids, same cold (10 iters) + warm (2 iters from cold converged
    centroids) regimes, same inertia/label-agreement gate, same timing
    helper. Backends under comparison: vibecuda (candidate) vs the
    configured cute baseline; PR23's triton backend runs as the diagnostic
    reference leg mirroring the upstream script's internal comparison.
    """
    # --- exact upstream input construction (seed 0, direct BF16 Gaussian) ---
    batch, samples, clusters, dim = 4, 8192, 64, 128
    iters, warm_iters = 10, 2
    torch.manual_seed(0)
    x = torch.randn(batch, samples, dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, samples, (batch, clusters), device=x.device)
    initial = torch.gather(
        x, 1, indices[..., None].expand(-1, -1, x.shape[-1])
    ).contiguous()

    def run(backend, n_iters=None, init=None):
        if backend == "vibecuda":
            return flashlib.flash_kmeans(
                x, clusters, max_iters=iters if n_iters is None else n_iters,
                tol=0.0, init_centroids=initial if init is None else init,
                backend="vibecuda")
        return batch_kmeans_Euclid(
            x, clusters, max_iters=iters if n_iters is None else n_iters,
            tol=0.0, init_centroids=initial if init is None else init,
            backend=backend)

    backends = ("vibecuda", "cute", "triton") if baseline_ok else ("vibecuda",)
    out_meta = [(torch.Size((batch, samples)), torch.int32, "int"),
                (torch.Size((batch, clusters, dim)), torch.bfloat16, "float")]

    # --- genuine prefill/full-write proof on every backend, untimed ---
    for backend in backends:
        if backend == "vibecuda":
            # Deterministic caller-buffer proof (no allocator heuristic).
            prove_full_write_vibecuda(x, initial, "lloyd", clusters, iters)
        else:
            # Immutable pinned sources: allocator-pool heuristic only.
            prove_full_write(lambda: run(backend), out_meta)

    # --- correctness gate (mirrors upstream triton-vs-cute check; here the
    # candidate must agree with the configured cute baseline AND with the
    # upstream triton reference leg) ---
    reference = {}
    for backend in backends:
        ids, centroids, _ = run(backend)
        torch.cuda.synchronize(device)
        reference[backend] = (ids, rel_inertia(x, centroids, ids))

    results = {"gate": {}, "cold": {}, "warm": {}, "baseline_ok": baseline_ok}
    if baseline_ok:
        base_inertia = reference["cute"][1]
        for backend in ("vibecuda", "triton"):
            rel = abs(reference[backend][1] - base_inertia) / abs(base_inertia)
            agree = label_agreement(reference[backend][0], reference["cute"][0])
            results["gate"][backend] = {"inertia_rel": rel,
                                        "label_agreement": agree,
                                        "pass": rel < rtol}
            if not rel < rtol:
                results["pass"] = False
                results["error"] = (f"{backend} inertia differs from cute by "
                                    f"{rel:.3e} (> {rtol:.1e})")
                return results
        del reference
        torch.cuda.empty_cache()
    else:
        # Candidate-only mode: invariant-preserving gate instead of
        # cross-backend comparisons. Mirror the multi-iteration Lloyd rows of
        # the fixed23 suite: gate relative inertia vs the exhaustive fp32
        # reference labels on the vibecuda final centroids (raw ids legitimately
        # diverge from any fp32 ref after 10 bf16 iterations, so an ids gate
        # would be miscalibrated). rtol=1e-2 as in the fixed bench.
        _, cents_v, _ = run("vibecuda")
        ref_ids = ref_assign_fp32(x, cents_v.to(x.dtype))
        ref_in = rel_inertia(x, cents_v, ref_ids)
        impl_in = reference["vibecuda"][1]
        rel = abs(impl_in - ref_in) / abs(ref_in) if ref_in else 0.0
        results["gate"]["vibecuda_vs_fp32_ref"] = {
            "inertia_rel": rel, "pass": rel < 1e-2,
            "label_agreement": label_agreement(
                reference["vibecuda"][0].to(torch.int32), ref_ids)}
        if not rel < 1e-2:
            results["pass"] = False
            results["error"] = (f"vibecuda inertia differs from fp32 ref by "
                                f"{rel:.3e} (> 1e-2)")
            return results
        del reference
        torch.cuda.empty_cache()

    # --- cold regime timing (exact upstream protocol) ---
    for backend in backends:
        results["cold"][backend] = pr23_cuda_time(
            lambda backend=backend: run(backend), warmup, repeats, device)
    # --- warm regime: restart from the cold solution's centroids ---
    _, warm_init, _ = run("cute" if baseline_ok else "vibecuda")
    torch.cuda.synchronize(device)
    for backend in backends:
        results["warm"][backend] = pr23_cuda_time(
            lambda backend=backend: run(backend, warm_iters, warm_init),
            warmup, repeats, device)

    if baseline_ok:
        results["speedups"] = {
            "cold_vibecuda_over_cute": results["cold"]["cute"] / results["cold"]["vibecuda"],
            "warm_vibecuda_over_cute": results["warm"]["cute"] / results["warm"]["vibecuda"],
            "cold_cute_over_triton": results["cold"]["triton"] / results["cold"]["cute"],
            "warm_cute_over_triton": results["warm"]["triton"] / results["warm"]["cute"],
        }
    else:
        results["speedups"] = None
    results["pass"] = True
    results["geometry"] = {"B": batch, "N": samples, "K": clusters, "D": dim,
                           "iters": iters, "warm_iters": warm_iters}
    return results


# ===========================================================================
# Suite 2: fixed 23-row task workload portfolio
# ===========================================================================

ROWS = [
    ("longcat_video_cold_b64_n37440_k293_i10", "lloyd", 64, 37440, 293, 10, 1, 0.0),
    ("longcat_video_warm_b64_n37440_k293_i2", "lloyd", 64, 37440, 293, 2, 2, 0.0),
    ("wan21_720p_query_b40_n75600_k300_i2", "lloyd", 40, 75600, 300, 2, 3, 0.0),
    ("wan21_720p_key_b40_n75600_k1000_i2", "lloyd", 40, 75600, 1000, 2, 4, 0.0),
    ("hunyuan_480p_query_b24_n44806_k200_i2", "lloyd", 24, 44806, 200, 2, 10, 0.0),
    ("hunyuan_480p_key_b24_n44806_k1000_i2", "lloyd", 24, 44806, 1000, 2, 11, 0.0),
    ("hunyuan_720p_query_b24_n119056_k400_i2", "lloyd", 24, 119056, 400, 2, 12, 0.0),
    ("hunyuan_720p_key_b24_n119056_k1000_i2", "lloyd", 24, 119056, 1000, 2, 13, 0.0),
    ("boundary_b2_n2_k2_i1", "lloyd", 2, 2, 2, 1, 20, 0.99),
    ("boundary_b2_n257_k3_i1", "lloyd", 2, 257, 3, 1, 21, 0.99),
    ("boundary_b2_n130_k128_i1", "lloyd", 2, 130, 128, 1, 22, 0.99),
    ("boundary_b2_n1024_k591_i1", "lloyd", 2, 1024, 591, 1, 23, 0.99),
    ("boundary_b2_n1024_k1024_i1", "lloyd", 2, 1024, 1024, 1, 24, 0.99),
] + [
    (f"predict_b2_n2048_k{k}", "predict", 2, 2048, k, 1, 100 + k, 0.99)
    for k in (2, 32, 64, 65, 96, 128, 129, 256, 591, 1024)
]


def make_inputs(batch, samples, clusters, seed, label):
    """Exact input generator from the authoritative workload table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(batch, samples, 128, generator=g, dtype=torch.float32,
                    device="cuda").to(torch.bfloat16)
    if label.startswith(("boundary_", "predict_")):
        idx = torch.stack([
            torch.randperm(samples, generator=g, device="cuda")[:clusters]
            for _ in range(batch)
        ])
    else:
        idx = torch.randint(0, samples, (batch, clusters), generator=g,
                            device="cuda")
    initial = torch.gather(x, 1, idx[..., None].expand(-1, -1, 128)).contiguous()
    return x, initial


def bench_call(fn, warmup=5, repeats=10):
    """FlashInfer CUPTI cold-L2 timing protocol on both backends."""
    from flashinfer.testing import bench_gpu_time

    with torch.inference_mode():
        times = bench_gpu_time(
            fn, enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False,
            dry_run_iters=warmup, repeat_iters=repeats,
        )
    return statistics.median(times)


def run_row(label, mode, B, N, K, iters, seed, agree_min, baseline_ok=True):
    x, initial = make_inputs(B, N, K, seed, label)
    out_meta = [(torch.Size((B, N)), torch.int32, "int")]
    if mode != "predict":
        # Predict returns the UNCHANGED input centroid tensor — not a
        # produced output, so it is excluded from the full-write proof.
        out_meta.append((torch.Size((B, K, 128)), torch.bfloat16, "float"))

    def cand():
        if mode == "predict":
            return flashlib.vibecuda_assign_euclid(x, initial), initial
        return flashlib.flash_kmeans(x, K, max_iters=iters, tol=0.0,
                                     init_centroids=initial,
                                     backend="vibecuda")[:2]

    def base():
        if mode == "predict":
            # Baseline's public assign-only path (fixed centroids, ids out).
            return euclid_assign(x, initial, None, backend="cute"), initial
        return batch_kmeans_Euclid(x, K, max_iters=iters, tol=0.0,
                                   init_centroids=initial, backend="cute")[:2]

    # Untimed full-write proof: deterministic caller-buffer proof for the
    # vibecuda candidate; allocator-pool heuristic for the immutable pinned
    # baseline (skipped in candidate-only mode).
    out_c = prove_full_write_vibecuda(x, initial, mode, K, iters)
    ids_c, cents_c = out_c[0], out_c[1]

    if baseline_ok:
        out_b = prove_full_write(base, out_meta)
        ids_b, cents_b = out_b[0], out_b[1]
        legs = (("vibecuda", ids_c, cents_c),
                ("kda_cute", ids_b, cents_b))
    else:
        legs = (("vibecuda", ids_c, cents_c),)

    checks = {}
    for name, ids, cents in legs:
        if mode == "lloyd" and iters > 1:
            # Multi-iteration rows are gated on relative inertia only.
            ref_ids = ref_assign_fp32(x, cents.to(x.dtype))
            ref_in = rel_inertia(x, cents, ref_ids)
            impl_in = rel_inertia(x, cents, ids)
            rel = abs(impl_in - ref_in) / abs(ref_in) if ref_in else 0.0
            checks[name] = {"inertia_rel": rel, "pass": rel <= 0.01}
        else:
            # 1-iteration Lloyd rows: ids come from the initial-centroid
            # assignment (update happens after the final assignment).
            ref_src = initial if mode == "predict" or iters == 1 else cents.to(x.dtype)
            ref_ids = ref_assign_fp32(x, ref_src)
            agree = label_agreement(ids.to(torch.int32), ref_ids)
            checks[name] = {"label_agreement": agree,
                            "pass": agree >= max(agree_min, 0.99)}
    t_c = bench_call(cand)
    if baseline_ok:
        cross = label_agreement(ids_c.to(torch.int32), ids_b.to(torch.int32))
        t_b = bench_call(base)
        row_pass = checks["vibecuda"]["pass"] and checks["kda_cute"]["pass"]
        speedup = t_b / t_c
    else:
        cross = None
        t_b = None
        row_pass = checks["vibecuda"]["pass"]
        speedup = None
    return {
        "label": label, "mode": mode, "B": B, "N": N, "K": K, "iters": iters,
        "candidate_ms": t_c, "baseline_ms": t_b, "speedup": speedup,
        "checks": checks, "cross_label_agreement": cross,
        "baseline_ok": baseline_ok,
        "pass": row_pass,
    }


def run_fixed23_suite(row_filter=None, baseline_ok=True):
    results = []
    for row in ROWS:
        if row_filter and row_filter not in row[0]:
            continue
        r = run_row(*row, baseline_ok=baseline_ok)
        results.append(r)
        print(json.dumps({k: r[k] for k in
                          ("label", "candidate_ms", "baseline_ms", "speedup",
                           "pass", "cross_label_agreement")}))

    sp = [r["speedup"] for r in results if r["speedup"] is not None]
    arith = sum(sp) / len(sp) if sp else None
    geo = 1.0
    for s in sp:
        geo *= s
    geo = geo ** (1.0 / len(sp)) if sp else None
    return {
        "workloads": len(results),
        "baseline_ok": baseline_ok,
        "all_pass": all(r["pass"] for r in results),
        "arithmetic_mean_speedup": arith,
        "geometric_mean_speedup": geo,
        "min_speedup": min(sp) if sp else None,
        "min_shape": (min(results, key=lambda r: r["speedup"])["label"]
                      if sp else None),
        "results": results,
    }


# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("both", "pr23", "fixed23"), default="both")
    p.add_argument("--out", default=None, help="write JSON results here")
    p.add_argument("--rows", default=None,
                   help="label substring filter (fixed23 suite only)")
    p.add_argument("--omit-baseline", action="store_true",
                   help="candidate-only mode: run every vibecuda leg "
                        "(full-write proofs, correctness gates, CUPTI timing) "
                        "without the pinned PR23 baseline; for sandboxes where "
                        "the baseline checkout is intentionally unavailable")
    args = p.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    baseline_ok = False
    if not args.omit_baseline:
        try:
            baseline_ok = bool(load_baseline())
        except Exception as exc:
            print(f"pinned PR23 baseline unavailable ({type(exc).__name__}: "
                  f"{exc}); falling back to candidate-only mode")

    print(f"flashlib from: {flashlib.__file__}")
    if baseline_ok:
        print("flash_kmeans (pinned PR23 baseline) from: "
              f"{_baseline_module.__file__}")
    else:
        print("flash_kmeans (pinned PR23 baseline): NOT AVAILABLE — "
              "candidate-only mode (speedup legs skip)")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)}")

    summary = {
        "baseline": ("flash_kmeans PR23 cute (svg-project/flash-kmeans#23@97ed5131)"
                     if baseline_ok else
                     "UNAVAILABLE in this process (pr23@97ed5131 pinned checkout "
                     "not importable here); candidate-only legs"),
        "baseline_ok": baseline_ok,
        "suites": {},
    }

    if args.suite in ("both", "pr23"):
        print("\n=== SUITE pr23 (exact benchmarks/benchmark_backends.py reproduction) ===")
        r = run_pr23_suite(device, baseline_ok=baseline_ok)
        summary["suites"]["pr23"] = r
        if r.get("pass"):
            print(json.dumps({
                "gate": r["gate"],
                "cold_ms": r["cold"], "warm_ms": r["warm"],
                "speedups": r["speedups"],
            }, indent=1))
        else:
            print(json.dumps(r, indent=1))

    if args.suite in ("both", "fixed23"):
        print("\n=== SUITE fixed23 (task-439 workload portfolio) ===")
        summary["suites"]["fixed23"] = run_fixed23_suite(
            args.rows, baseline_ok=baseline_ok)
        s = summary["suites"]["fixed23"]
        print(json.dumps({k: v for k, v in s.items() if k != "results"}, indent=1))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    suites = summary["suites"]
    ok = (("pr23" not in suites or suites["pr23"].get("pass"))
          and ("fixed23" not in suites or suites["fixed23"]["all_pass"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
