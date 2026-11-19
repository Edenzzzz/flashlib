// KDA Lloyd clustering kernels: BF16 D=128 fixed-centroid assignment and
// iterative Lloyd assign/update, pure custom CUDA (no cuBLAS, no Triton,
// no CuTe/CUTLASS templates).
//
// Semantics matched to the task-level reference:
//   assign: ids[n] = argmin_k ( |c_k|^2 - 2 x_n . c_k ), first-min-index wins.
//           (row constant |x_n|^2 dropped; argmin identical.)
//   update: c_k <- sum_{n: id=k} x_n / count_k (fp32 accum), keep previous
//           centroid row when count_k == 0, result rounded to BF16.
// The assign computes x.c with bf16 tensor-core MMA and f32 accumulation,
// folds val = acc - 0.5*|c|^2 with a strict-< update in ascending K order so
// first-index tie-breaking matches the reference (modulo fp32 reduction
// order, tolerated by the label-agreement / relative-inertia gates).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <vector>
#include <cstdint>

#define DK 128      // feature dim (fixed by contract)
#define BM 128      // point rows per assign CTA
#define KC 64       // centroid rows per smem chunk
#define CPW 136     // smem centroid row pitch in bf16 elems (pad: conflict spread)
#define THREADS 128

using bf16 = __nv_bfloat16;

static inline const bf16* as_bf16(const at::BFloat16* p) {
  return reinterpret_cast<const bf16*>(p);
}
static inline bf16* as_bf16(at::BFloat16* p) {
  return reinterpret_cast<bf16*>(p);
}

// warp-level bf16 MMA: c[4] += a x b, m16n8k16 row.col f32.bf16.bf16.f32
// a[0]: (row g,   k 2q+{0,1})   a[1]: (row g+8, k 2q+{0,1})
// a[2]: (row g,   k 2q+8+{0,1}) a[3]: (row g+8, k 2q+8+{0,1})  g=lane/4 q=lane%4
// b[0]: (k 2q+{0,1}, n=g)       b[1]: (k 2q+8+{0,1}, n=g)
// c[0..1]: (row g, n 2q+{0,1})  c[2..3]: (row g+8, n 2q+{0,1})
static __device__ __forceinline__ void mma_bf16_16816(
    float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Programmatic dependent launch (PDL): every kernel in this file is launched
// through pdl_launch (cudaLaunchAttributeProgrammaticStreamSerialization), so
// griddepcontrol ops are always legal. Discipline: pdl_fire() (launch
// dependents) sits strictly AFTER the kernel's last global store, so the
// dependent grid's pdl_wait() observes complete predecessor state under both
// the grid-completion and the trigger-scope readings of griddepcontrol.wait;
// pdl_wait() sits BEFORE the first global read of any buffer the predecessor
// could have written. Loads of loop-invariant inputs (X) may be issued above
// pdl_wait to overlap the fetch with the predecessor tail.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_fire() {
  asm volatile("griddepcontrol.launch_dependents;");
}

// hcnorm: hc[row] = 0.5 * |c_row|^2 in fp32. one warp per centroid row.
__global__ void hcnorm_kernel(const bf16* __restrict__ C, float* __restrict__ hc,
                              int64_t rows) {
  pdl_wait();  // PDL: predecessor writes visible here
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const bf16* c = C + row * DK;
  int lane = threadIdx.x;
  uint2 pk = reinterpret_cast<const uint2*>(c)[lane];
  const bf16* e = reinterpret_cast<const bf16*>(&pk);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v = __bfloat162float(e[i]);
    s += v * v;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) hc[row] = 0.5f * s;
  pdl_fire();  // PDL: all global stores above are complete
}

// assignment kernel.
// grid (ceil(N/BM), B); block 128 = 4 warps; warp owns two 16-row tiles
// (tile rows warp*32 + t*16 + 0..15 within the CTA tile).
// assignment kernel.
// grid (ceil(N/BM) * KSPLIT, B); block 128 = 4 warps; warp owns two 16-row tiles
// (tile rows warp*32 + t*16 + 0..15 within the CTA tile).
// KSPLIT > 1 splits the centroid chunks across CTAs: blockIdx.x / ntiles is
// the split index, each split covers chunks [s*CH_PER_SPLIT, ...). Partials go
// to PV/PI [(b*KSPLIT + s)*N + r]; merge_assign_kernel combines them. With
// PV == nullptr the split must be full-range and IDS is written directly.
// When SC is non-null the (tile 0, split 0) CTA of each batch also zeroes its
// batch's SC slice: that lets the host drop the separate sc.zero_() fill on
// the small-update path. The zeroing overlaps the other CTAs' compute and is
// stream-ordered before the later scatter_small/finalize launches, so no
// extra synchronization is needed.
__global__ void __launch_bounds__(THREADS) assign_kernel(
    const bf16* __restrict__ X, const bf16* __restrict__ C,
    const float* __restrict__ HC, int32_t* __restrict__ IDS, int N, int K,
    int ntiles, int ksplits, int chunks_per_split, float* __restrict__ PV,
    int32_t* __restrict__ PI, float* __restrict__ SC) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ uint32_t smem_raw[];
  constexpr int CPW_U32 = CPW / 2;  // 68
  uint32_t* smemC = smem_raw;                                    // KC x CPW_U32
  float* smemHC = reinterpret_cast<float*>(smem_raw + KC * CPW_U32);

  const int b = blockIdx.y;
  const int tile = blockIdx.x % ntiles;
  const int split = blockIdx.x / ntiles;
  if (SC != nullptr && tile == 0 && split == 0) {
    float* scb = SC + (int64_t)b * K * 129;
    for (int i = threadIdx.x; i < K * 129; i += THREADS) scb[i] = 0.f;
  }
  const int64_t m0 = (int64_t)tile * BM;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = lane >> 2;
  const int qq = lane & 3;

  // load this warp's 32 A rows into fragments (clamp invalid rows to N-1;
  // their results are never stored)
  uint32_t A[2][8][4];
  {
    const bf16* xb = X + (int64_t)b * N * DK;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int r = (int)(m0 + warp * 32 + t * 16);
      int ra = min(r + grp, N - 1);
      int rb = min(r + grp + 8, N - 1);
      const bf16* pa = xb + (int64_t)ra * DK;
      const bf16* pb = xb + (int64_t)rb * DK;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int kk = ks * 16 + qq * 2;
        A[t][ks][0] = *reinterpret_cast<const uint32_t*>(pa + kk);
        A[t][ks][1] = *reinterpret_cast<const uint32_t*>(pb + kk);
        A[t][ks][2] = *reinterpret_cast<const uint32_t*>(pa + kk + 8);
        A[t][ks][3] = *reinterpret_cast<const uint32_t*>(pb + kk + 8);
      }
    }
  }

  float bv[2][2];  // per tile t, row grp + 8*h
  int bi[2][2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    bv[t][0] = bv[t][1] = CUDART_INF_F;
    bi[t][0] = bi[t][1] = 0;
  }

  const bf16* cb = C + (int64_t)b * K * DK;
  // HC is no longer read here: the chunk's hc values are computed in-kernel
  // (see the fused hcnorm block in the staging loop below).
  const int chunks = (K + KC - 1) / KC;
  const int ch_beg = split * chunks_per_split;
  const int ch_end = min(ch_beg + chunks_per_split, chunks);

  for (int ch = ch_beg; ch < ch_end; ++ch) {
    const int k0 = ch * KC;
    {  // stage centroid chunk to smem with zero tail pad
      constexpr int U32_PER_ROW = DK / 2;
#pragma unroll 4
      for (int i = threadIdx.x; i < KC * U32_PER_ROW; i += THREADS) {
        const int n = i / U32_PER_ROW;
        const int k2 = i - n * U32_PER_ROW;
        uint32_t v = 0;
        if (k0 + n < K)
          v = reinterpret_cast<const uint32_t*>(cb + (int64_t)(k0 + n) * DK)[k2];
        smemC[n * CPW_U32 + k2] = v;
      }
      // Fused hcnorm for this chunk (replaces the standalone hcnorm_kernel
      // launch on the small-N path). Loads the same uint2 per lane from
      // global C with the identical fp32 add/shfl order as hcnorm_kernel, so
      // the computed values are bitwise-identical to the removed launch; the
      // (g >= K) -> INF pad reproduces the old consumer-side guard. These
      // loads hit the same lines the smemC staging above already pulls into
      // L2/L1 and are independent of the staging writes, so no extra barrier
      // is needed beyond the existing __syncthreads below.
      for (int n = warp; n < KC; n += THREADS / 32) {
        const int g = k0 + n;
        float s = 0.f;
        if (g < K) {
          const bf16* c = cb + (int64_t)g * DK;
          const uint2 pk = reinterpret_cast<const uint2*>(c)[lane];
          const bf16* e = reinterpret_cast<const bf16*>(&pk);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float v = __bfloat162float(e[i]);
            s += v * v;
          }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) smemHC[n] = (g < K) ? 0.5f * s : CUDART_INF_F;
      }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < 8; ++m) {
      // Narrow-K specialization: a subtile whose first row k0 + m*8 reaches K
      // is entirely zero-staged centroid rows with hc == INF, so its fold
      // yields v == INF and the strict-< update below can never move the
      // running best. Exiting early is bitwise-identical (first-index
      // tie-breaking is untouched) and skips up to 7/8 of the MMA + argmin
      // work on narrow-K or tail chunks (K <= 32 -> 4 subtiles; K = 293 tail
      // chunk with 37 valid rows -> 5 subtiles). The break condition depends
      // only on (k0, K), so it is uniform across the block.
      if (k0 + m * 8 >= K) break;
      const float hc0 = smemHC[m * 8 + qq * 2];
      const float hc1 = smemHC[m * 8 + qq * 2 + 1];
      float acc[2][4];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
      }
      const uint32_t* brow = smemC + (m * 8 + grp) * CPW_U32;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t bb[2] = {brow[ks * 8 + qq], brow[ks * 8 + qq + 4]};
        mma_bf16_16816(acc[0], A[0][ks], bb);
        mma_bf16_16816(acc[1], A[1][ks], bb);
      }
      // fold: rows grp (acc[t][0..1]) and grp+8 (acc[t][2..3])
#pragma unroll
      for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float v0 = hc0 - acc[t][2 * h];
          float v1 = hc1 - acc[t][2 * h + 1];
          int c0 = k0 + m * 8 + qq * 2;
          bool lt = (v1 < v0);
          float v = lt ? v1 : v0;
          int c = lt ? (c0 + 1) : c0;
#pragma unroll
          for (int o = 1; o <= 2; o <<= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, v, o);
            int oc = __shfl_xor_sync(0xffffffffu, c, o);
            if (ov < v || (ov == v && oc < c)) {
              v = ov;
              c = oc;
            }
          }
          if (v < bv[t][h]) {
            bv[t][h] = v;
            bi[t][h] = c;
          }
        }
      }
    }
    __syncthreads();
  }

  if (qq != 0) return;
  if (PV != nullptr) {
    float* pvb = PV + ((int64_t)b * ksplits + split) * N;
    int32_t* pib = PI + ((int64_t)b * ksplits + split) * N;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int r = (int)(m0 + warp * 32 + t * 16) + grp;
      if (r < N) {
        pvb[r] = bv[t][0];
        pib[r] = bi[t][0];
      }
      if (r + 8 < N) {
        pvb[r + 8] = bv[t][1];
        pib[r + 8] = bi[t][1];
      }
    }
    pdl_fire();  // PDL: all global stores above are complete
    return;
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    int r = (int)(m0 + warp * 32 + t * 16) + grp;
    if (r < N) IDS[(int64_t)b * N + r] = bi[t][0];
    if (r + 8 < N) IDS[(int64_t)b * N + r + 8] = bi[t][1];
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// merge split-K assign partials. One thread per row; splits ascend in k so a
// strict-< update keeps the reference's first-min-index tie-break across
// equal values (earlier splits always carry smaller k ranges).
__global__ void merge_assign_kernel(const float* __restrict__ PV,
                                    const int32_t* __restrict__ PI,
                                    int32_t* __restrict__ IDS, int64_t N,
                                    int ksplits) {
  pdl_wait();  // PDL: predecessor writes visible here
  const int b = blockIdx.y;
  const int64_t r = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (r >= N) return;
  const float* pvb = PV + (int64_t)b * ksplits * N + r;
  const int32_t* pib = PI + (int64_t)b * ksplits * N + r;
  float bv = CUDART_INF_F;
  int bi = 0;
  for (int s = 0; s < ksplits; ++s) {
    const float v = pvb[(int64_t)s * N];
    if (v < bv) {
      bv = v;
      bi = pib[(int64_t)s * N];
    }
  }
  IDS[(int64_t)b * N + r] = bi;
  pdl_fire();  // PDL: all global stores above are complete
}

// ---------------------------------------------------------------------------
// tcgen05 (UMMA) assignment path for large N.
//
// CTA tile: BM=256 X rows x BN=256 centroids, full K=128 reduction as 8 UMMA
// m128n256k16 steps. Two M=128 halves accumulate into TMEM cols [0,256) and
// [256,512). B (centroid) chunks are double-buffered with cp.async into
// 128B-swizzled K-major SMEM (atom = 8 rows x 64 bf16 = 1024 B; row-block
// stride 2048 B since each row is 256 B). Thread t of 256 owns X row t and
// drains its accumulator row directly: lane = TMEM lane = row-in-half, so the
// epilogue needs no cross-thread reduction. Ascending-k strict-< fold into
// (val, idx) keeps the reference's first-min-index tie-break.
// ---------------------------------------------------------------------------

// (file is compiled only for sm_100a/sm_103a, so tcgen05 is always available)
#define UBM 256      // X rows per CTA
#define UBN 256      // centroids per chunk
#define UTHREADS 256
#define UMAXK 1024

__device__ __forceinline__ uint32_t swz_addr(uint32_t base, int r, int k) {
  // K-major bf16, 128B swizzle, 256B rows. k must be a multiple of 8 (16B unit).
  return base + (r >> 3) * 2048 + ((k >> 6) << 10) + (r & 7) * 128 +
         ((((k & 63) >> 3) ^ (r & 7)) << 4);
}

__device__ __forceinline__ uint64_t make_umma_desc(uint64_t mask, uint32_t addr_bytes) {
  return mask | ((uint64_t)(addr_bytes & 0x3FFFF) >> 4);
}

// swizzle 128B (01), LBO unused->1, SBO = 2048B >> 4 = 128
#define UMMA_DESC_MASK ((1ULL << 62) | (1ULL << 16) | (128ULL << 32))

__device__ __forceinline__ uint32_t elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred %%px;\n\telect.sync _|%%px, %1;\n\t@%%px mov.s32 %0, 1;\n\t}"
      : "+r"(pred)
      : "r"(0xFFFFFFFFu));
  return pred;
}

__device__ __forceinline__ void umma_f16(uint32_t tmem_c, uint64_t da, uint64_t db,
                                         uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
      :
      : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit(uint32_t mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(mbar));
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  // one poll, then a short sleep to avoid burning issue slots while the MMA
  // drains (the spin was ~11% of all executed instructions), then spin.
  uint32_t done = 0;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}"
      : "=r"(done)
      : "r"(mbar), "r"(phase));
  if (done) return;
  asm volatile("nanosleep.u32 32;");
  while (!done) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
  }
}

__device__ __forceinline__ void mbar_arrive(uint32_t mbar) {
  uint64_t tok;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(tok) : "r"(mbar));
}

// arrive on mbar when ALL prior cp.async ops of this thread complete; with
// .noinc the arrivals count against the barrier's init count.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t mbar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(mbar));
}

__device__ __forceinline__ void tmem_ld32(float r[32], uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]),
        "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]),
        "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]), "=f"(r[17]),
        "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]),
        "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]),
        "=f"(r[30]), "=f"(r[31])
      : "r"(taddr));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
               "r"(src_bytes));
}

// Fold one 32-column block of accumulator rows into (bv, bi).
// Interleaved (value, index) tournament: level 1 fuses with the packed subs
// (mv = hc - r via sub.f32x2, materializing 16 winner (v,i) pairs with
// immediate indices), levels 2..5 merge in place with one predicate + two
// selects per node. Fully converged: the previous index-free/recovery scheme
// (value-only tree + on-improvement bit-mask + __ffs) was divergent — the
// improvement branch fires warp-level on nearly every chunk (union over 32
// lanes) and serialized ~80 SASS ops per fold, 15-26% of assign time.
// Tie-break keeps the LOWEST index on equal values (strict < update), matching
// the reference's first-min-index rule.
// argmin over 32 columns; hc points at 16 float2 (this thread's chunk window).
__device__ __forceinline__ uint64_t subq_f32x2(uint64_t a, uint64_t b) {
  uint64_t d;
  asm("sub.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ void argmin32(const float r[32], const float2 hc[16],
                                         int kbase, float& bv, int& bi) {
  float v[16];
  int ix[16];
#pragma unroll
  for (int q = 0; q < 16; ++q) {
    const uint64_t d = subq_f32x2(*reinterpret_cast<const uint64_t*>(&hc[q]),
                                  *reinterpret_cast<const uint64_t*>(&r[2 * q]));
    const float2 dv = *reinterpret_cast<const float2*>(&d);
    v[q] = fminf(dv.x, dv.y);
    ix[q] = 2 * q + ((dv.y < dv.x) ? 1 : 0);
  }
#pragma unroll
  for (int st = 1; st < 16; st <<= 1) {
#pragma unroll
    for (int q = 0; q + st < 16; q += 2 * st) {
      const bool p = v[q + st] < v[q];
      v[q] = fminf(v[q], v[q + st]);
      ix[q] = p ? ix[q + st] : ix[q];
    }
  }
  if (v[0] < bv) {
    bv = v[0];
    bi = kbase + ix[0];
  }
}

// K-tail partial drain/fold for the final chunk of a partial-K centroid
// sweep. Only the last chunk can overhang K; dead columns inside it are
// provably non-winning (their B rows were zero-filled by cp.async and their
// HC entries INF-padded), so a warp whose 32-col window extends past K may
// drain and fold just the valid prefix. On K=293 warps 4-7 fold 5 columns
// instead of 32 on the tail chunk, cutting dead issue slots from the
// st_math-bound convoy. np = ceil(nv/2) valid float2 pairs; np <= 8.
__device__ __forceinline__ void tmem_ld8(float r[8], uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]),
        "=f"(r[6]), "=f"(r[7])
      : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld16(float r[16], uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]),
        "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]),
        "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15])
      : "r"(taddr));
}

// NP is a template parameter: every array index is compile-time so the whole
// fold stays in registers (a runtime np bound spills the arrays to the stack
// and cost ~10% end-to-end). The switch on np at the call site is
// warp-uniform, so there is no divergence.
template <int NP>
__device__ __forceinline__ void argminN(const float* r, const float4* hcp,
                                        int kbase, float& bv, int& bi) {
  float v[8];
  int ix[8];
  const float2* hc = reinterpret_cast<const float2*>(hcp);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    if (q < NP) {
      const uint64_t d =
          subq_f32x2(*reinterpret_cast<const uint64_t*>(&hc[q]),
                     *reinterpret_cast<const uint64_t*>(&r[2 * q]));
      const float2 dv = *reinterpret_cast<const float2*>(&d);
      v[q] = fminf(dv.x, dv.y);
      ix[q] = 2 * q + ((dv.y < dv.x) ? 1 : 0);
    } else {
      v[q] = CUDART_INF_F;
      ix[q] = 2 * q;
    }
  }
#pragma unroll
  for (int st = 1; st < 8; st <<= 1) {
#pragma unroll
    for (int q = 0; q + st < 8; q += 2 * st) {
      const bool p = v[q + st] < v[q];
      v[q] = fminf(v[q], v[q + st]);
      ix[q] = p ? ix[q + st] : ix[q];
    }
  }
  if (v[0] < bv) {  // same strict-< first-min-index tie-break as argmin32
    bv = v[0];
    bi = kbase + ix[0];
  }
}

// Drain (2*NP cols of each M-half via one tcgen05.ld) + fold for the tail
// chunk's partial window. hc loads stay in registers: NP is compile-time.
template <int NP>
__device__ __forceinline__ void fold_tail(uint32_t t0, uint32_t t1,
                                          const float* hc_kbase, int kbase,
                                          float& bv0, int& bi0, float& bv1,
                                          int& bi1) {
  float s0[2 * NP], s1[2 * NP];
  if (NP > 4) {
    tmem_ld16(s0, t0);
    tmem_ld16(s1, t1);
  } else {
    tmem_ld8(s0, t0);
    tmem_ld8(s1, t1);
  }
  const float4* hcp = reinterpret_cast<const float4*>(hc_kbase);
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  argminN<NP>(s0, hcp, kbase, bv0, bi0);
  argminN<NP>(s1, hcp, kbase, bv1, bi1);
  asm volatile("tcgen05.fence::before_thread_sync;");
}


// idesc: dense f32 acc, bf16 A/B, no transpose, M=128 (M>>4=8), N=256 (N>>3=32)
#define UMMA_IDESC \
  ((1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(UBN >> 3) << 17) | ((uint32_t)(128 >> 4) << 24))


// ---------------------------------------------------------------------------
// 3-CTA/SM software-pipelined tcgen05 assign: BM=128, BN=64 chunks, two B
// stages (2 x 16 KB). 69.7 KB smem + 128 TMEM cols per CTA -> 3 CTAs resident
// per SM (209 KB smem, 384 TMEM cols). Steady state at chunk c:
//   wait MMA(c) -> prefetch B(c+2) into the freed stage -> drain TMEM(c-1? no,
//   cols[c&1]) while MMA(c+1) (issued at the end of this iteration? no -
//   issued at iteration c) overlaps -> issue MMA(c+1) after load(c+1) lands.
// Thread t owns X row t (t<128): lane = TMEM lane = row; the fold uses a
// register tree argmin (strict < keeps the lower k on ties -> reference's
// first-min-index tie-break).
// ---------------------------------------------------------------------------
#define UBN4 64
// idesc: dense f32 acc, bf16 A/B, no transpose, M=128, N=64 (N>>3=8)
#define UMMA_IDESC64 \
  ((1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) | ((uint32_t)(128 >> 4) << 24))

// ---------------------------------------------------------------------------
// BM=256 variant: two M=128 UMMA halves share one B pipeline. Halves the CTA
// count (fixed per-CTA costs: prologue, per-chunk barriers, tmem alloc, hc
// staging) at 2 CTAs/SM. TMEM: 4 x 64-col buffers (half h, stage s at col
// (h*2+s)*64). Thread t folds its TMEM lane for both halves, so the per-chunk
// per-thread fold doubles while the barrier round-trip count per row halves.
// Register budget: 2 CTAs/SM -> up to ~128 regs allowed (launch_bounds 256,2).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) assign_umma2x_cp_kernel(
    const bf16* __restrict__ X, const bf16* __restrict__ C,
    const float* __restrict__ HC, int32_t* __restrict__ IDS, int N, int K) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t a_sm = (uint32_t)__cvta_generic_to_shared(smem);          // 64 KB
  const uint32_t b0_sm = a_sm + 65536;                                     // 16 KB stage 0
  const uint32_t b1_sm = b0_sm + 16384;                                    // 16 KB stage 1
  const uint32_t ext = 65536 + 16384 * 2;                                  // 96 KB for A+B
  float* hc_sm = reinterpret_cast<float*>(smem + ext);                     // 4 KB
  uint32_t* aux = reinterpret_cast<uint32_t*>(smem + ext + 4096);          // [0]=tmem
  float* mv_sm = reinterpret_cast<float*>(smem + ext + 4096 + 16);         // 2x128 f32
  int* mi_sm = reinterpret_cast<int*>(smem + ext + 4096 + 16 + 1024);      // 2x128 i32
  const uint32_t mbar0 = a_sm + ext + 4096 + 16 + 2048;
  const uint32_t mbar1 = mbar0 + 8;

  const int b = blockIdx.y;
  const int m0 = blockIdx.x * 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;      // 0..7; warps w and w+4 share lanes 32*(w&3)
  const int lane = tid & 31;
  const int nchunks = (K + UBN4 - 1) / UBN4;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar1));
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            a_sm + ext + 4096),
        "r"(256));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  for (int i = tid; i < nchunks * UBN4; i += 256)
    hc_sm[i] = (i < K) ? HC[(int64_t)b * K + i] : CUDART_INF_F;

  const bf16* cb = C + (int64_t)b * K * DK;
  // sliding B-chunk prefetch state (structure shared with the BM=256 x_cp kernel below)
  const int pb_r = tid >> 4;
  const int pb_off = (tid & 15) * 8;
  uint32_t pb_doff[4];
  const bf16* pb_ptr[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int row = pb_r + 16 * m;
    pb_doff[m] = swz_addr(0, row, pb_off);
    pb_ptr[m] = cb + (int64_t)min(row, K - 1) * DK + pb_off;
  }
  // A tile (256 rows) + B chunks 0..1 (both stages)
  {
    const bf16* xb = X + (int64_t)b * N * DK;
#pragma unroll 4
    for (int u = tid; u < 256 * (DK / 8); u += 256) {
      const int r = u >> 4, unit = u & 15;
      const int rr = min(m0 + r, N - 1);
      cp_async16(swz_addr(a_sm, r, unit * 8), xb + (int64_t)rr * DK + unit * 8, 16);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int row = pb_r + 16 * m;
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], (row < K) ? 16u : 0u);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int row = pb_r + 16 * m;
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m] + UBN4 * DK,
                 (UBN4 + row < K) ? 16u : 0u);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) pb_ptr[m] += 2 * UBN4 * DK;
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
  const uint32_t tmem = aux[0];

  if (warp < 2) {  // warps 0/1 issue MMA(0) for their own M-half (disjoint
    // TMEM windows, each chain self-ordered; commit barrier count is 2)
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (elect_one()) {
      const int h = warp;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
        const uint64_t da = make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
        umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                 UMMA_IDESC64, ks > 0);
      }
      umma_commit(mbar0);
    }
  }

  float bv0 = CUDART_INF_F, bv1 = CUDART_INF_F;
  int bi0 = 0, bi1 = 0;
  const uint32_t lane_base = (uint32_t)(((warp & 3) * 32)) << 16;
  const int coff = (warp >> 2) * 32;  // column offset within the 64-col chunk

  // Bulk/tail split: in the bulk phase the prefetch target (chunk c+2) is a
  // full 64-row chunk and chunks c+1/c+2 always exist, so the K-guard
  // predicates and the two existence branches are statically true. The
  // guarded body below is retained verbatim for the last few chunks.
  int c = 0;
  int cbulk = K / UBN4 - 2;
  if (cbulk > nchunks) cbulk = nchunks;
  if (cbulk < 0) cbulk = 0;
#pragma unroll 2
  for (; c < cbulk; ++c) {
    mbar_wait((c & 1) ? mbar1 : mbar0, (c >> 1) & 1);
    float r0[32], r1[32];
    {
      const uint32_t tbase = tmem + (c & 1) * 64 + coff;
      tmem_ld32(r0, tbase + lane_base);
      tmem_ld32(r1, tbase + 128 + lane_base);
    }
    {
      const uint32_t bdst = (c & 1) ? b1_sm : b0_sm;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        cp_async16(bdst + pb_doff[m], pb_ptr[m], 16u);
        pb_ptr[m] += UBN4 * DK;
      }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    // no fence.proxy.async here: in-loop smem writes are all async-proxy
    // (cp.async); wait_group + __syncthreads already order cp.async writes
    // before the UMMA reads (async<->async). The all-thread fence was paying
    // membar stalls with nothing to publish (measured st_membar hot in NCU).
    if (warp < 2) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const uint32_t bs = ((c + 1) & 1) ? b1_sm : b0_sm;
        const uint32_t tb = tmem + ((c + 1) & 1) * 64;
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tb + h * 128, da, make_umma_desc(UMMA_DESC_MASK, bs + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit((c + 1) & 1 ? mbar1 : mbar0);
      }
    }
    {
      const int kbase = c * UBN4 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }
  for (; c < nchunks; ++c) {
    mbar_wait((c & 1) ? mbar1 : mbar0, (c >> 1) & 1);
    // K-tail crop: only the final chunk can overhang K (nv < 32). Dead
    // columns are provably non-winning (zero-filled B rows, INF-padded HC),
    // so a warp whose 32-col window overhangs drains/folds just the valid
    // prefix (nv <= 16) or skips entirely (nv <= 0); 16 < nv < 32 keeps the
    // full 32-wide fold since most columns are valid anyway.
    const int nv = K - c * UBN4 - coff;
    float r0[32], r1[32];
    const uint32_t tbase = tmem + (c & 1) * 64 + coff;
    if (nv > 16) {  // full or near-full window: identical dataflow to before
      // async TMEM drains for both halves (different column windows); issue
      // FIRST so their return latency overlaps the prefetch issue below.
      tmem_ld32(r0, tbase + lane_base);
      tmem_ld32(r1, tbase + 128 + lane_base);
    }
    if (c + 2 < nchunks) {
      const int k0 = (c + 2) * UBN4;
      const uint32_t bdst = (c & 1) ? b1_sm : b0_sm;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int row = pb_r + 16 * m;
        cp_async16(bdst + pb_doff[m], pb_ptr[m], (k0 + row < K) ? 16u : 0u);
        pb_ptr[m] += UBN4 * DK;
      }
    }
    asm volatile("cp.async.commit_group;");  // keep wait_group 1 accounting uniform
    if (c + 1 < nchunks) {
      asm volatile("cp.async.wait_group 1;");
      __syncthreads();
      // no fence.proxy.async (see bulk loop): only async-proxy smem writes here
      if (warp < 2) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (elect_one()) {
          const uint32_t bs = ((c + 1) & 1) ? b1_sm : b0_sm;
          const uint32_t tb = tmem + ((c + 1) & 1) * 64;
          const int h = warp;
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
            const uint64_t da =
                make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
            umma_f16(tb + h * 128, da, make_umma_desc(UMMA_DESC_MASK, bs + off),
                     UMMA_IDESC64, ks > 0);
          }
          umma_commit((c + 1) & 1 ? mbar1 : mbar0);
        }
      }
    }
    if (nv > 16) {
      const int kbase = c * UBN4 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    } else if (nv > 0) {
      // partial window: drain and fold only the valid prefix (np pairs).
      // Warp-uniform switch over compile-time NP keeps every index static.
      const int np = (nv + 1) >> 1;
      const int kbase = c * UBN4 + coff;
      const uint32_t t0 = tbase + lane_base;
      const uint32_t t1 = tbase + 128 + lane_base;
      switch (np) {
        case 1: fold_tail<1>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 2: fold_tail<2>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 3: fold_tail<3>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 4: fold_tail<4>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 5: fold_tail<5>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 6: fold_tail<6>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 7: fold_tail<7>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        default: fold_tail<8>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
      }
    }
    // nv <= 0: window entirely past K -> no drain, no fold
  }

  // merge col-halves for both M-halves (warps 4-7 -> smem, 0-3 reduce)
  if (warp >= 4) {
    const int slot = (warp - 4) * 32 + lane;
    mv_sm[slot] = bv0;
    mi_sm[slot] = bi0;
    mv_sm[128 + slot] = bv1;
    mi_sm[128 + slot] = bi1;
  }
  __syncthreads();
  if (warp < 4) {
    const int r = warp * 32 + lane;
    {
      const float ov = mv_sm[r];
      const int oc = mi_sm[r];
      if (ov < bv0 || (ov == bv0 && oc < bi0)) {
        bv0 = ov;
        bi0 = oc;
      }
      const int row = m0 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi0;
    }
    {
      const float ov = mv_sm[128 + r];
      const int oc = mi_sm[128 + r];
      if (ov < bv1 || (ov == bv1 && oc < bi1)) {
        bv1 = ov;
        bi1 = oc;
      }
      const int row = m0 + 128 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi1;
    }
  }

  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(256));
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// BM=256 DEC variant: as assign_umma2x_cp_kernel but with per-B-stage fill
// mbarriers (cp.async.mbarrier.arrive.noinc) gating MMA issue instead of the
// whole-CTA commit_group+wait_group+__syncthreads convoy; MMA(c+1) issues as
// soon as the stage fill completes, shrinking loop-head mbar_wait blocking
// on filled grids (measured on sm100: wan21_720p_key assign +2.9% / full
// +2.8%, longcat_cold assign +4.3%, bitwise-identical outputs; underfilled
// grids keep the base kernel via host dispatch).
// BM=256 variant: two M=128 UMMA halves share one B pipeline. Halves the CTA
// count (fixed per-CTA costs: prologue, per-chunk barriers, tmem alloc, hc
// staging) at 2 CTAs/SM. TMEM: 4 x 64-col buffers (half h, stage s at col
// (h*2+s)*64). Thread t folds its TMEM lane for both halves, so the per-chunk
// per-thread fold doubles while the barrier round-trip count per row halves.
// Register budget: 2 CTAs/SM -> up to ~128 regs allowed (launch_bounds 256,2).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) assign_umma2x_dec_kernel(
    const bf16* __restrict__ X, const bf16* __restrict__ C,
    const float* __restrict__ HC, int32_t* __restrict__ IDS, int N, int K) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t a_sm = (uint32_t)__cvta_generic_to_shared(smem);          // 64 KB
  const uint32_t b0_sm = a_sm + 65536;                                     // 16 KB stage 0
  const uint32_t b1_sm = b0_sm + 16384;                                    // 16 KB stage 1
  const uint32_t ext = 65536 + 16384 * 2;                                  // 96 KB for A+B
  float* hc_sm = reinterpret_cast<float*>(smem + ext);                     // 4 KB
  uint32_t* aux = reinterpret_cast<uint32_t*>(smem + ext + 4096);          // [0]=tmem
  float* mv_sm = reinterpret_cast<float*>(smem + ext + 4096 + 16);         // 2x128 f32
  int* mi_sm = reinterpret_cast<int*>(smem + ext + 4096 + 16 + 1024);      // 2x128 i32
  const uint32_t mbar0 = a_sm + ext + 4096 + 16 + 2048;
  const uint32_t mbar1 = mbar0 + 8;
  const uint32_t bmbar0 = mbar0 + 16;
  const uint32_t bmbar1 = bmbar0 + 8;

  const int b = blockIdx.y;
  const int m0 = blockIdx.x * 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;      // 0..7; warps w and w+4 share lanes 32*(w&3)
  const int lane = tid & 31;
  const int nchunks = (K + UBN4 - 1) / UBN4;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar1));
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            a_sm + ext + 4096),
        "r"(256));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();  // dec: barrier inits + tmem aux visible before arrivals

  for (int i = tid; i < nchunks * UBN4; i += 256)
    hc_sm[i] = (i < K) ? HC[(int64_t)b * K + i] : CUDART_INF_F;

  const bf16* cb = C + (int64_t)b * K * DK;
  // sliding B-chunk prefetch state (structure shared with the BM=256 x_cp kernel below)
  const int pb_r = tid >> 4;
  const int pb_off = (tid & 15) * 8;
  uint32_t pb_doff[4];
  const bf16* pb_ptr[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int row = pb_r + 16 * m;
    pb_doff[m] = swz_addr(0, row, pb_off);
    pb_ptr[m] = cb + (int64_t)min(row, K - 1) * DK + pb_off;
  }
  // A tile (256 rows) + B chunks 0..1 (both stages)
  {
    const bf16* xb = X + (int64_t)b * N * DK;
#pragma unroll 4
    for (int u = tid; u < 256 * (DK / 8); u += 256) {
      const int r = u >> 4, unit = u & 15;
      const int rr = min(m0 + r, N - 1);
      cp_async16(swz_addr(a_sm, r, unit * 8), xb + (int64_t)rr * DK + unit * 8, 16);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int row = pb_r + 16 * m;
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], (row < K) ? 16u : 0u);
    }
    cp_async_mbar_arrive_noinc(bmbar0);  // covers A+B0 completion (this thread)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int row = pb_r + 16 * m;
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m] + UBN4 * DK,
                 (UBN4 + row < K) ? 16u : 0u);
    }
    cp_async_mbar_arrive_noinc(bmbar1);
#pragma unroll
    for (int m = 0; m < 4; ++m) pb_ptr[m] += 2 * UBN4 * DK;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
  const uint32_t tmem = aux[0];

  if (warp < 2) {  // warps 0/1 issue MMA(0) for their own M-half (disjoint
    // TMEM windows, each chain self-ordered; commit barrier count is 2)
    mbar_wait(bmbar0, 0);  // dec: stage-0 fill (A+B0) complete
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (elect_one()) {
      const int h = warp;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
        const uint64_t da = make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
        umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                 UMMA_IDESC64, ks > 0);
      }
      umma_commit(mbar0);
    }
  }

  float bv0 = CUDART_INF_F, bv1 = CUDART_INF_F;
  int bi0 = 0, bi1 = 0;
  const uint32_t lane_base = (uint32_t)(((warp & 3) * 32)) << 16;
  const int coff = (warp >> 2) * 32;  // column offset within the 64-col chunk

  // Bulk/tail split: in the bulk phase the prefetch target (chunk c+2) is a
  // full 64-row chunk and chunks c+1/c+2 always exist, so the K-guard
  // predicates and the two existence branches are statically true. The
  // guarded body below is retained verbatim for the last few chunks.
  int c = 0;
  int cbulk = K / UBN4 - 2;
  if (cbulk > nchunks) cbulk = nchunks;
  if (cbulk < 0) cbulk = 0;
#pragma unroll 2
  for (; c < cbulk; ++c) {
    mbar_wait((c & 1) ? mbar1 : mbar0, (c >> 1) & 1);
    float r0[32], r1[32];
    {
      const uint32_t tbase = tmem + (c & 1) * 64 + coff;
      tmem_ld32(r0, tbase + lane_base);
      tmem_ld32(r1, tbase + 128 + lane_base);
    }
    {
      const uint32_t bdst = (c & 1) ? b1_sm : b0_sm;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        cp_async16(bdst + pb_doff[m], pb_ptr[m], 16u);
        pb_ptr[m] += UBN4 * DK;
      }
    }
    cp_async_mbar_arrive_noinc((c & 1) ? bmbar1 : bmbar0);  // dec: fill c+2
    if (warp < 2) {
      mbar_wait(((c + 1) & 1) ? bmbar1 : bmbar0, ((c + 1) >> 1) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const uint32_t bs = ((c + 1) & 1) ? b1_sm : b0_sm;
        const uint32_t tb = tmem + ((c + 1) & 1) * 64;
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tb + h * 128, da, make_umma_desc(UMMA_DESC_MASK, bs + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit((c + 1) & 1 ? mbar1 : mbar0);
      }
    }
    {
      const int kbase = c * UBN4 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }
  for (; c < nchunks; ++c) {
    mbar_wait((c & 1) ? mbar1 : mbar0, (c >> 1) & 1);
    // K-tail crop: only the final chunk can overhang K (nv < 32). Dead
    // columns are provably non-winning (zero-filled B rows, INF-padded HC),
    // so a warp whose 32-col window overhangs drains/folds just the valid
    // prefix (nv <= 16) or skips entirely (nv <= 0); 16 < nv < 32 keeps the
    // full 32-wide fold since most columns are valid anyway.
    const int nv = K - c * UBN4 - coff;
    float r0[32], r1[32];
    const uint32_t tbase = tmem + (c & 1) * 64 + coff;
    if (nv > 16) {  // full or near-full window: identical dataflow to before
      // async TMEM drains for both halves (different column windows); issue
      // FIRST so their return latency overlaps the prefetch issue below.
      tmem_ld32(r0, tbase + lane_base);
      tmem_ld32(r1, tbase + 128 + lane_base);
    }
    if (c + 2 < nchunks) {
      const int k0 = (c + 2) * UBN4;
      const uint32_t bdst = (c & 1) ? b1_sm : b0_sm;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int row = pb_r + 16 * m;
        cp_async16(bdst + pb_doff[m], pb_ptr[m], (k0 + row < K) ? 16u : 0u);
        pb_ptr[m] += UBN4 * DK;
      }
      cp_async_mbar_arrive_noinc((c & 1) ? bmbar1 : bmbar0);
    }
    if (c + 1 < nchunks) {
      if (warp < 2) {
        mbar_wait(((c + 1) & 1) ? bmbar1 : bmbar0, ((c + 1) >> 1) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (elect_one()) {
          const uint32_t bs = ((c + 1) & 1) ? b1_sm : b0_sm;
          const uint32_t tb = tmem + ((c + 1) & 1) * 64;
          const int h = warp;
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
            const uint64_t da =
                make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
            umma_f16(tb + h * 128, da, make_umma_desc(UMMA_DESC_MASK, bs + off),
                     UMMA_IDESC64, ks > 0);
          }
          umma_commit((c + 1) & 1 ? mbar1 : mbar0);
        }
      }
    }
    if (nv > 16) {
      const int kbase = c * UBN4 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    } else if (nv > 0) {
      // partial window: drain and fold only the valid prefix (np pairs).
      // Warp-uniform switch over compile-time NP keeps every index static.
      const int np = (nv + 1) >> 1;
      const int kbase = c * UBN4 + coff;
      const uint32_t t0 = tbase + lane_base;
      const uint32_t t1 = tbase + 128 + lane_base;
      switch (np) {
        case 1: fold_tail<1>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 2: fold_tail<2>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 3: fold_tail<3>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 4: fold_tail<4>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 5: fold_tail<5>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 6: fold_tail<6>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        case 7: fold_tail<7>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
        default: fold_tail<8>(t0, t1, hc_sm + kbase, kbase, bv0, bi0, bv1, bi1); break;
      }
    }
    // nv <= 0: window entirely past K -> no drain, no fold
  }

  // merge col-halves for both M-halves (warps 4-7 -> smem, 0-3 reduce)
  if (warp >= 4) {
    const int slot = (warp - 4) * 32 + lane;
    mv_sm[slot] = bv0;
    mi_sm[slot] = bi0;
    mv_sm[128 + slot] = bv1;
    mi_sm[128 + slot] = bi1;
  }
  __syncthreads();
  if (warp < 4) {
    const int r = warp * 32 + lane;
    {
      const float ov = mv_sm[r];
      const int oc = mi_sm[r];
      if (ov < bv0 || (ov == bv0 && oc < bi0)) {
        bv0 = ov;
        bi0 = oc;
      }
      const int row = m0 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi0;
    }
    {
      const float ov = mv_sm[128 + r];
      const int oc = mi_sm[128 + r];
      if (ov < bv1 || (ov == bv1 && oc < bi1)) {
        bv1 = ov;
        bi1 = oc;
      }
      const int row = m0 + 128 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi1;
    }
  }

  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(256));
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// ---------------------------------------------------------------------------
// K=293 compile-time specialization of assign_umma2x_dec_kernel (longcat
// rows). nchunks=5, cbulk=2; chunk 4 has 37 valid rows, so coff=0 warps see a
// full 32-col window and coff=32 warps fold exactly 5 cols (fold_tail<3>).
// All dynamic chunk-existence tests, K-guard predicates, tail-window branches
// and the fold_tail np-switch of the generic kernel are resolved at compile
// time; the barrier/mbarrier structure, phases, stage parity and fold
// tie-breaks are identical. Dispatch gates this kernel on K==293 only.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) assign_umma2x_dec_k293_kernel(
    const bf16* __restrict__ X, const bf16* __restrict__ C,
    const float* __restrict__ HC, int32_t* __restrict__ IDS, int N, int K) {
  (void)K;  // contractually 293 (host-dispatch gated); signature kept uniform
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t a_sm = (uint32_t)__cvta_generic_to_shared(smem);          // 64 KB
  const uint32_t b0_sm = a_sm + 65536;                                     // 16 KB stage 0
  const uint32_t b1_sm = b0_sm + 16384;                                    // 16 KB stage 1
  const uint32_t ext = 65536 + 16384 * 2;                                  // 96 KB for A+B
  float* hc_sm = reinterpret_cast<float*>(smem + ext);                     // 4 KB
  uint32_t* aux = reinterpret_cast<uint32_t*>(smem + ext + 4096);          // [0]=tmem
  float* mv_sm = reinterpret_cast<float*>(smem + ext + 4096 + 16);         // 2x128 f32
  int* mi_sm = reinterpret_cast<int*>(smem + ext + 4096 + 16 + 1024);      // 2x128 i32
  const uint32_t mbar0 = a_sm + ext + 4096 + 16 + 2048;
  const uint32_t mbar1 = mbar0 + 8;
  const uint32_t bmbar0 = mbar0 + 16;
  const uint32_t bmbar1 = bmbar0 + 8;

  const int b = blockIdx.y;
  const int m0 = blockIdx.x * 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;      // 0..7; warps w and w+4 share lanes 32*(w&3)
  const int lane = tid & 31;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar1));
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            a_sm + ext + 4096),
        "r"(256));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();  // barrier inits + tmem aux visible before arrivals

  // PDL A-burst hoist: X is a loop-invariant input across the whole Lloyd
  // run, so the full 64KB A tile issues BEFORE the dependency wait and the
  // fetch overlaps the predecessor (hcnorm / previous finalize) tail. All
  // reads of C (rewritten in place by finalize each iteration) and HC
  // (hcnorm output) stay strictly behind pdl_wait. The per-thread cp.async
  // tracking sets are unchanged: bmbar0 still covers A(16)+B0(4), bmbar1
  // B1(4); the mbar phases fold identically.
  {
    const bf16* xb = X + (int64_t)b * N * DK;
#pragma unroll 4
    for (int u = tid; u < 256 * (DK / 8); u += 256) {
      const int r = u >> 4, unit = u & 15;
      const int rr = min(m0 + r, N - 1);
      cp_async16(swz_addr(a_sm, r, unit * 8), xb + (int64_t)rr * DK + unit * 8, 16);
    }
  }
  pdl_wait();  // C (finalize) and HC (hcnorm) become visible here

  for (int i = tid; i < 320; i += 256)
    hc_sm[i] = (i < 293) ? HC[(int64_t)b * 293 + i] : CUDART_INF_F;

  const bf16* cb = C + (int64_t)b * 293 * DK;
  const int pb_r = tid >> 4;
  const int pb_off = (tid & 15) * 8;
  uint32_t pb_doff[4];
  const bf16* pb_ptr[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int row = pb_r + 16 * m;
    pb_doff[m] = swz_addr(0, row, pb_off);
    pb_ptr[m] = cb + (int64_t)row * DK + pb_off;  // row <= 63 < 293: no clamp
  }
  // B chunks 0..1 (both stages); all rows < 128 <= K so every cp.async is a
  // full 16B load (no partial-row predicates); the clamp analysis is the
  // same as the A burst above.
  {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], 16u);
    }
    cp_async_mbar_arrive_noinc(bmbar0);  // covers A+B0 completion (this thread)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m] + UBN4 * DK, 16u);
    }
    cp_async_mbar_arrive_noinc(bmbar1);
#pragma unroll
    for (int m = 0; m < 4; ++m) pb_ptr[m] += 2 * UBN4 * DK;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
  const uint32_t tmem = aux[0];

  if (warp < 2) {  // MMA(0) -> stage 0 (identical to the generic kernel)
    mbar_wait(bmbar0, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (elect_one()) {
      const int h = warp;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
        const uint64_t da = make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
        umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                 UMMA_IDESC64, ks > 0);
      }
      umma_commit(mbar0);
    }
  }

  float bv0 = CUDART_INF_F, bv1 = CUDART_INF_F;
  int bi0 = 0, bi1 = 0;
  const uint32_t lane_base = (uint32_t)(((warp & 3) * 32)) << 16;
  const int coff = (warp >> 2) * 32;  // column offset within the 64-col chunk

  // ---- c=0 (stage 0): full fold, prefetch chunk 2 -> b0, MMA(1) ----
  mbar_wait(mbar0, 0);  // MMA(0) complete
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + coff + lane_base);
    tmem_ld32(r1, tmem + 128 + coff + lane_base);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], 16u);  // chunk 2: rows 128..191 < 293
      pb_ptr[m] += UBN4 * DK;
    }
    cp_async_mbar_arrive_noinc(bmbar0);  // fill of chunk 2
    if (warp < 2) {
      mbar_wait(bmbar1, 0);  // chunk-1 fill (prologue) complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + 64 + h * 128, da,
                   make_umma_desc(UMMA_DESC_MASK, b1_sm + off), UMMA_IDESC64,
                   ks > 0);
        }
        umma_commit(mbar1);  // MMA(1) -> stage 1
      }
    }
    {
      const int kbase = coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=1 (stage 1): full fold, prefetch chunk 3 -> b1, MMA(2) ----
  mbar_wait(mbar1, 0);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + 64 + coff + lane_base);
    tmem_ld32(r1, tmem + 64 + 128 + coff + lane_base);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m], 16u);  // chunk 3: rows 192..255 < 293
      pb_ptr[m] += UBN4 * DK;
    }
    cp_async_mbar_arrive_noinc(bmbar1);  // fill of chunk 3
    if (warp < 2) {
      mbar_wait(bmbar0, 1);  // chunk-2 fill complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit(mbar0);  // MMA(2) -> stage 0
      }
    }
    {
      const int kbase = 64 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=2 (stage 0): full fold, prefetch chunk 4 (37 valid rows), MMA(3) ----
  mbar_wait(mbar0, 1);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + coff + lane_base);
    tmem_ld32(r1, tmem + 128 + coff + lane_base);
    // chunk 4: global rows 256..292 valid; rows 293..319 zero-filled so dead
    // columns stay non-winning (same zfill semantics as the generic kernel).
    cp_async16(b0_sm + pb_doff[0], pb_ptr[0], 16u);
    cp_async16(b0_sm + pb_doff[1], pb_ptr[1], 16u);
    cp_async16(b0_sm + pb_doff[2], pb_ptr[2], (pb_r < 5) ? 16u : 0u);
    cp_async16(b0_sm + pb_doff[3], pb_ptr[3], 0u);
    cp_async_mbar_arrive_noinc(bmbar0);  // fill of chunk 4
    if (warp < 2) {
      mbar_wait(bmbar1, 1);  // chunk-3 fill complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + 64 + h * 128, da,
                   make_umma_desc(UMMA_DESC_MASK, b1_sm + off), UMMA_IDESC64,
                   ks > 0);
        }
        umma_commit(mbar1);  // MMA(3) -> stage 1
      }
    }
    {
      const int kbase = 128 + coff;  // cols 128..191: all < 293, full window
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=3 (stage 1): full fold, no prefetch, MMA(4) ----
  mbar_wait(mbar1, 1);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + 64 + coff + lane_base);
    tmem_ld32(r1, tmem + 64 + 128 + coff + lane_base);
    if (warp < 2) {
      mbar_wait(bmbar0, 0);  // chunk-4 fill complete (phase parity back to 0)
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit(mbar0);  // MMA(4) -> stage 0
      }
    }
    {
      const int kbase = 192 + coff;  // cols 192..255: all < 293, full window
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=4 (stage 0, tail): warps 0-3 fold cols 256..287 (full 32-wide),
  // warps 4-7 fold the 5 valid cols 288..292 (same split as the generic
  // kernel's nv=37 / nv=5 tail path) ----
  mbar_wait(mbar0, 0);
  if (coff == 0) {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + lane_base);
    tmem_ld32(r1, tmem + 128 + lane_base);
    const int kbase = 256;
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    float2 hv[16];
    const float4* hcp = reinterpret_cast<const float4*>(hc_sm + 256);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4 f = hcp[q];
      hv[2 * q] = make_float2(f.x, f.y);
      hv[2 * q + 1] = make_float2(f.z, f.w);
    }
    argmin32(r0, hv, kbase, bv0, bi0);
    argmin32(r1, hv, kbase, bv1, bi1);
    asm volatile("tcgen05.fence::before_thread_sync;");
  } else {
    fold_tail<3>(tmem + 32 + lane_base, tmem + 128 + 32 + lane_base,
                 hc_sm + 288, 288, bv0, bi0, bv1, bi1);
  }

  // merge col-halves for both M-halves (warps 4-7 -> smem, 0-3 reduce)
  if (warp >= 4) {
    const int slot = (warp - 4) * 32 + lane;
    mv_sm[slot] = bv0;
    mi_sm[slot] = bi0;
    mv_sm[128 + slot] = bv1;
    mi_sm[128 + slot] = bi1;
  }
  __syncthreads();
  if (warp < 4) {
    const int r = warp * 32 + lane;
    {
      const float ov = mv_sm[r];
      const int oc = mi_sm[r];
      if (ov < bv0 || (ov == bv0 && oc < bi0)) {
        bv0 = ov;
        bi0 = oc;
      }
      const int row = m0 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi0;
    }
    {
      const float ov = mv_sm[128 + r];
      const int oc = mi_sm[128 + r];
      if (ov < bv1 || (ov == bv1 && oc < bi1)) {
        bv1 = ov;
        bi1 = oc;
      }
      const int row = m0 + 128 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi1;
    }
  }

  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(256));
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// ---------------------------------------------------------------------------
// K=300 compile-time specialization of assign_umma2x_dec_kernel (wan21_720p_
// query row: B=40, N=75600 — decoupled grid is always filled). nchunks=5,
// cbulk=2; chunk 4 has 44 valid rows, so coff=0 warps see a full 32-col
// window and coff=32 warps fold exactly 12 cols (fold_tail<6>). Same rewrite
// as the k293 spec: dynamic chunk-existence tests, K-guard predicates,
// tail-window branches and the fold_tail np-switch resolved at compile time;
// barrier/mbarrier structure, phases, stage parity and fold tie-breaks
// identical. Dispatch gates this kernel on K==300 only.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) assign_umma2x_dec_k300_kernel(
    const bf16* __restrict__ X, const bf16* __restrict__ C,
    const float* __restrict__ HC, int32_t* __restrict__ IDS, int N, int K) {
  (void)K;  // contractually 300 (host-dispatch gated); signature kept uniform
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t a_sm = (uint32_t)__cvta_generic_to_shared(smem);          // 64 KB
  const uint32_t b0_sm = a_sm + 65536;                                     // 16 KB stage 0
  const uint32_t b1_sm = b0_sm + 16384;                                    // 16 KB stage 1
  const uint32_t ext = 65536 + 16384 * 2;                                  // 96 KB for A+B
  float* hc_sm = reinterpret_cast<float*>(smem + ext);                     // 4 KB
  uint32_t* aux = reinterpret_cast<uint32_t*>(smem + ext + 4096);          // [0]=tmem
  float* mv_sm = reinterpret_cast<float*>(smem + ext + 4096 + 16);         // 2x128 f32
  int* mi_sm = reinterpret_cast<int*>(smem + ext + 4096 + 16 + 1024);      // 2x128 i32
  const uint32_t mbar0 = a_sm + ext + 4096 + 16 + 2048;
  const uint32_t mbar1 = mbar0 + 8;
  const uint32_t bmbar0 = mbar0 + 16;
  const uint32_t bmbar1 = bmbar0 + 8;

  const int b = blockIdx.y;
  const int m0 = blockIdx.x * 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;      // 0..7; warps w and w+4 share lanes 32*(w&3)
  const int lane = tid & 31;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(mbar1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(bmbar1));
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            a_sm + ext + 4096),
        "r"(256));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();  // barrier inits + tmem aux visible before arrivals

  for (int i = tid; i < 320; i += 256)
    hc_sm[i] = (i < 300) ? HC[(int64_t)b * 300 + i] : CUDART_INF_F;

  const bf16* cb = C + (int64_t)b * 300 * DK;
  const int pb_r = tid >> 4;
  const int pb_off = (tid & 15) * 8;
  uint32_t pb_doff[4];
  const bf16* pb_ptr[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int row = pb_r + 16 * m;
    pb_doff[m] = swz_addr(0, row, pb_off);
    pb_ptr[m] = cb + (int64_t)row * DK + pb_off;  // row <= 63 < 300: no clamp
  }
  // A tile (256 rows) + B chunks 0..1 (both stages); all rows < 128 <= K so
  // every cp.async is a full 16B load (no partial-row predicates).
  {
    const bf16* xb = X + (int64_t)b * N * DK;
#pragma unroll 4
    for (int u = tid; u < 256 * (DK / 8); u += 256) {
      const int r = u >> 4, unit = u & 15;
      const int rr = min(m0 + r, N - 1);
      cp_async16(swz_addr(a_sm, r, unit * 8), xb + (int64_t)rr * DK + unit * 8, 16);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], 16u);
    }
    cp_async_mbar_arrive_noinc(bmbar0);  // covers A+B0 completion (this thread)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m] + UBN4 * DK, 16u);
    }
    cp_async_mbar_arrive_noinc(bmbar1);
#pragma unroll
    for (int m = 0; m < 4; ++m) pb_ptr[m] += 2 * UBN4 * DK;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
  const uint32_t tmem = aux[0];

  if (warp < 2) {  // MMA(0) -> stage 0 (identical to the generic kernel)
    mbar_wait(bmbar0, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (elect_one()) {
      const int h = warp;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
        const uint64_t da = make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
        umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                 UMMA_IDESC64, ks > 0);
      }
      umma_commit(mbar0);
    }
  }

  float bv0 = CUDART_INF_F, bv1 = CUDART_INF_F;
  int bi0 = 0, bi1 = 0;
  const uint32_t lane_base = (uint32_t)(((warp & 3) * 32)) << 16;
  const int coff = (warp >> 2) * 32;  // column offset within the 64-col chunk

  // ---- c=0 (stage 0): full fold, prefetch chunk 2 -> b0, MMA(1) ----
  mbar_wait(mbar0, 0);  // MMA(0) complete
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + coff + lane_base);
    tmem_ld32(r1, tmem + 128 + coff + lane_base);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b0_sm + pb_doff[m], pb_ptr[m], 16u);  // chunk 2: rows 128..191 < 300
      pb_ptr[m] += UBN4 * DK;
    }
    cp_async_mbar_arrive_noinc(bmbar0);  // fill of chunk 2
    if (warp < 2) {
      mbar_wait(bmbar1, 0);  // chunk-1 fill (prologue) complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + 64 + h * 128, da,
                   make_umma_desc(UMMA_DESC_MASK, b1_sm + off), UMMA_IDESC64,
                   ks > 0);
        }
        umma_commit(mbar1);  // MMA(1) -> stage 1
      }
    }
    {
      const int kbase = coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=1 (stage 1): full fold, prefetch chunk 3 -> b1, MMA(2) ----
  mbar_wait(mbar1, 0);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + 64 + coff + lane_base);
    tmem_ld32(r1, tmem + 64 + 128 + coff + lane_base);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      cp_async16(b1_sm + pb_doff[m], pb_ptr[m], 16u);  // chunk 3: rows 192..255 < 300
      pb_ptr[m] += UBN4 * DK;
    }
    cp_async_mbar_arrive_noinc(bmbar1);  // fill of chunk 3
    if (warp < 2) {
      mbar_wait(bmbar0, 1);  // chunk-2 fill complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit(mbar0);  // MMA(2) -> stage 0
      }
    }
    {
      const int kbase = 64 + coff;
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=2 (stage 0): full fold, prefetch chunk 4 (44 valid rows), MMA(3) ----
  mbar_wait(mbar0, 1);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + coff + lane_base);
    tmem_ld32(r1, tmem + 128 + coff + lane_base);
    // chunk 4: global rows 256..299 valid; rows 300..319 zero-filled so dead
    // columns stay non-winning (same zfill semantics as the generic kernel).
    cp_async16(b0_sm + pb_doff[0], pb_ptr[0], 16u);
    cp_async16(b0_sm + pb_doff[1], pb_ptr[1], 16u);
    cp_async16(b0_sm + pb_doff[2], pb_ptr[2], (pb_r < 12) ? 16u : 0u);
    cp_async16(b0_sm + pb_doff[3], pb_ptr[3], 0u);
    cp_async_mbar_arrive_noinc(bmbar0);  // fill of chunk 4
    if (warp < 2) {
      mbar_wait(bmbar1, 1);  // chunk-3 fill complete
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + 64 + h * 128, da,
                   make_umma_desc(UMMA_DESC_MASK, b1_sm + off), UMMA_IDESC64,
                   ks > 0);
        }
        umma_commit(mbar1);  // MMA(3) -> stage 1
      }
    }
    {
      const int kbase = 128 + coff;  // cols 128..191: all < 300, full window
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=3 (stage 1): full fold, no prefetch, MMA(4) ----
  mbar_wait(mbar1, 1);
  {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + 64 + coff + lane_base);
    tmem_ld32(r1, tmem + 64 + 128 + coff + lane_base);
    if (warp < 2) {
      mbar_wait(bmbar0, 0);  // chunk-4 fill complete (phase parity back to 0)
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (elect_one()) {
        const int h = warp;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const uint32_t off = (ks >> 2) * 1024 + (ks & 3) * 32;
          const uint64_t da =
              make_umma_desc(UMMA_DESC_MASK, a_sm + 32768 * h + off);
          umma_f16(tmem + h * 128, da, make_umma_desc(UMMA_DESC_MASK, b0_sm + off),
                   UMMA_IDESC64, ks > 0);
        }
        umma_commit(mbar0);  // MMA(4) -> stage 0
      }
    }
    {
      const int kbase = 192 + coff;  // cols 192..255: all < 300, full window
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      float2 hv[16];
      const float4* hcp = reinterpret_cast<const float4*>(hc_sm + kbase);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 f = hcp[q];
        hv[2 * q] = make_float2(f.x, f.y);
        hv[2 * q + 1] = make_float2(f.z, f.w);
      }
      argmin32(r0, hv, kbase, bv0, bi0);
      argmin32(r1, hv, kbase, bv1, bi1);
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }

  // ---- c=4 (stage 0, tail): warps 0-3 fold cols 256..287 (full 32-wide),
  // warps 4-7 fold the 12 valid cols 288..299 (same split as the generic
  // kernel's nv=44 / nv=12 tail path) ----
  mbar_wait(mbar0, 0);
  if (coff == 0) {
    float r0[32], r1[32];
    tmem_ld32(r0, tmem + lane_base);
    tmem_ld32(r1, tmem + 128 + lane_base);
    const int kbase = 256;
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    float2 hv[16];
    const float4* hcp = reinterpret_cast<const float4*>(hc_sm + 256);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4 f = hcp[q];
      hv[2 * q] = make_float2(f.x, f.y);
      hv[2 * q + 1] = make_float2(f.z, f.w);
    }
    argmin32(r0, hv, kbase, bv0, bi0);
    argmin32(r1, hv, kbase, bv1, bi1);
    asm volatile("tcgen05.fence::before_thread_sync;");
  } else {
    fold_tail<6>(tmem + 32 + lane_base, tmem + 128 + 32 + lane_base,
                 hc_sm + 288, 288, bv0, bi0, bv1, bi1);
  }

  // merge col-halves for both M-halves (warps 4-7 -> smem, 0-3 reduce)
  if (warp >= 4) {
    const int slot = (warp - 4) * 32 + lane;
    mv_sm[slot] = bv0;
    mi_sm[slot] = bi0;
    mv_sm[128 + slot] = bv1;
    mi_sm[128 + slot] = bi1;
  }
  __syncthreads();
  if (warp < 4) {
    const int r = warp * 32 + lane;
    {
      const float ov = mv_sm[r];
      const int oc = mi_sm[r];
      if (ov < bv0 || (ov == bv0 && oc < bi0)) {
        bv0 = ov;
        bi0 = oc;
      }
      const int row = m0 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi0;
    }
    {
      const float ov = mv_sm[128 + r];
      const int oc = mi_sm[128 + r];
      if (ov < bv1 || (ov == bv1 && oc < bi1)) {
        bv1 = ov;
        bi1 = oc;
      }
      const int row = m0 + 128 + r;
      if (row < N) IDS[(int64_t)b * N + row] = bi1;
    }
  }

  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(256));
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// per-chunk histogram: sh counts -> CHIST[b, chunk, k]
__global__ void hist_kernel(const int32_t* __restrict__ IDS,
                            int32_t* __restrict__ CHIST, int64_t N, int K,
                            int chunk, int nchunks) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ int sh[];  // K ints
  for (int i = threadIdx.x; i < K; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const int b = blockIdx.y;
  const int64_t p0 = (int64_t)blockIdx.x * chunk;
  const int64_t pend = min(p0 + chunk, N);
  const int32_t* idb = IDS + (int64_t)b * N;
  for (int64_t p = p0 + threadIdx.x; p < pend; p += blockDim.x)
    atomicAdd(&sh[idb[p]], 1);
  __syncthreads();
  int32_t* dst = CHIST + ((int64_t)b * nchunks + blockIdx.x) * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) dst[i] = sh[i];
  pdl_fire();  // PDL: all global stores above are complete
}

// one block (1024 threads) per batch: sums per-chunk histograms, scans the
// totals, writes OFF[k], CHUNK_OFF[k], SC[...,128] counts, and rewrites the
// histogram buffer IN PLACE into per-chunk placement cursors
// HIST[b, c, k] = OFF[k] + sum_{c'<c} chist[c'][k].
// Requires K <= 1024.
//
// In-place safety: column k is owned by thread t=k alone. That thread reads
// every chunk slot in the count loop (or from the smem staging copy) before
// the cursor loop runs, and inside the cursor loop each slot's value is read
// before the same thread's store to that same slot, in strictly increasing
// chunk order — no slot is ever read after being overwritten.
__global__ void plan_kernel(int32_t* HIST,
                            int32_t* __restrict__ OFF,
                            int32_t* __restrict__ CHUNK_OFF,
                            float* __restrict__ SC, int K, int C, int nchunks,
                            int staged, int2* __restrict__ CIDMAP,
                            int64_t rchunks, const bf16* __restrict__ Cold,
                            bf16* __restrict__ Cnew, float* __restrict__ HC,
                            int32_t* __restrict__ DONEC) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ int smc[];  // staged CHIST copy when staged=1 (nchunks*K)
  const int b = blockIdx.x;
  const int t = threadIdx.x;
  const int64_t cb = (int64_t)b * nchunks * K;
  const int32_t* chp = HIST + cb;
  if (staged) {
    // coalesced staging pass: after this all chunk loops read smem, removing
    // the serial L2-latency chains that dominated this kernel
    for (int i = t; i < nchunks * K; i += blockDim.x) smc[i] = HIST[cb + i];
    __syncthreads();
    chp = smc;
  }
  int cnt = 0;
  if (t < K) {
    // 4 independent chains keep MLP high on the global path; with staging
    // these read smem and the latency vanishes
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int c = 0;
    for (; c + 3 < nchunks; c += 4) {
      c0 += chp[(int64_t)c * K + t];
      c1 += chp[(int64_t)(c + 1) * K + t];
      c2 += chp[(int64_t)(c + 2) * K + t];
      c3 += chp[(int64_t)(c + 3) * K + t];
    }
    for (; c < nchunks; ++c) c0 += chp[(int64_t)c * K + t];
    cnt = c0 + c1 + c2 + c3;
  }
  // one packed 64-bit block scan replaces two Hillis-Steele passes (40
  // syncthreads -> 4): high 32 bits = count, low 32 = ceil(count/C); a single
  // prefix-sum pass yields both the OFF and CHUNK_OFF scans. Counts are
  // <= N (~120k) and chunks <= N/C+1, so the packing is exact.
  const int ch = (t < K) ? (cnt + C - 1) / C : 0;
  unsigned long long v = ((unsigned long long)(unsigned)cnt << 32) | (unsigned)ch;
  const int lane = t & 31;
  const int w = t >> 5;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long u = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += u;
  }
  __shared__ unsigned long long ws[32];
  if (lane == 31) ws[w] = v;
  __syncthreads();
  if (w == 0) {
    unsigned long long u = ws[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long p = __shfl_up_sync(0xffffffffu, u, o);
      if (lane >= o) u += p;
    }
    ws[lane] = u;
  }
  __syncthreads();
  if (w > 0) v += ws[w - 1];
  const int incl = (int)(v >> 32);
  const int excl = incl - cnt;  // exclusive prefix of total counts
  const int cincl = (int)(unsigned)v;
  if (t < K) {
    OFF[b * (K + 1) + t] = excl;
    if (t == K - 1) OFF[b * (K + 1) + K] = incl;
    SC[((int64_t)b * K + t) * 129 + DK] = (float)cnt;
    // per-chunk cursors written in place over the histogram: every hN value
    // is read before its slot's store, and slots are touched in increasing
    // chunk order (in-place safety argument at the kernel comment)
    int run = excl;
    int c = 0;
    for (; c + 3 < nchunks; c += 4) {
      const int h0 = chp[(int64_t)(c + 0) * K + t];
      const int h1 = chp[(int64_t)(c + 1) * K + t];
      const int h2 = chp[(int64_t)(c + 2) * K + t];
      const int h3 = chp[(int64_t)(c + 3) * K + t];
      HIST[cb + (int64_t)(c + 0) * K + t] = run; run += h0;
      HIST[cb + (int64_t)(c + 1) * K + t] = run; run += h1;
      HIST[cb + (int64_t)(c + 2) * K + t] = run; run += h2;
      HIST[cb + (int64_t)(c + 3) * K + t] = run; run += h3;
    }
    for (; c < nchunks; ++c) {
      const int hv = chp[(int64_t)c * K + t];
      HIST[cb + (int64_t)c * K + t] = run; run += hv;
    }
    CHUNK_OFF[b * (K + 1) + t] = cincl - ch;
    if (t == K - 1) CHUNK_OFF[b * (K + 1) + K] = cincl;
    // cid -> (k, s0 base) map for reduce: kills its per-block binary search
    if (CIDMAP != nullptr) {
      const int coff = cincl - ch;
      for (int j = 0; j < ch; ++j) {
        CIDMAP[b * rchunks + coff + j] = make_int2(t, excl + j * C);
      }
    }
    // per-label reduce-chunk budget for the fused finalize: reduce CTAs of
    // label t decrement this once each after their SC accumulation; the
    // decrementer that observes 1 is the label's final chunk and performs
    // the centroid/HC finalization inline. Rewritten every iteration, so no
    // zeroing pass is needed.
    DONEC[(int64_t)b * K + t] = ch;
  }
  // Empty-label fallback for the fused finalize: count-0 labels own no
  // reduce chunk, so no reduce CTA ever finalizes them. plan already knows
  // every label's count here; after the barrier the counts are block-visible,
  // each warp ballots its 32 labels and copies the retained old centroid row
  // (bit-exact 8B copy) plus 0.5|row|^2 for the empty ones — the work the
  // standalone finalize_kernel did for count==0 rows. The common case
  // ballots to zero and costs one strided CNT load per label.
  __syncthreads();
  {
    const int wrp = t >> 5, ln = t & 31;
    for (int kb = wrp * 32; kb < K; kb += (int)(blockDim.x >> 5) * 32) {
      const int j = kb + ln;
      unsigned m = __ballot_sync(
          0xffffffffu, j < K && SC[((int64_t)b * K + j) * 129 + DK] == 0.f);
      while (m) {
        const int e = __ffs(m) - 1;
        m &= m - 1;
        const int lbl = kb + e;
        const uint2 pk =
            reinterpret_cast<const uint2*>(Cold + ((int64_t)b * K + lbl) * DK)[ln];
        reinterpret_cast<uint2*>(Cnew + ((int64_t)b * K + lbl) * DK)[ln] = pk;
        const bf16* ev = reinterpret_cast<const bf16*>(&pk);
        float ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const float v = __bfloat162float(ev[u]);
          ss += v * v;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1)
          ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (ln == 0) HC[(int64_t)b * K + lbl] = 0.5f * ss;
      }
    }
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// placement: per-chunk atomic cursor over per-chunk label bases (CBASE).
// ORDT is uint16_t when N <= 65535 (ord only stores global point ids < N).
template <typename ORDT>
__global__ void place_kernel(const int32_t* __restrict__ IDS,
                             int32_t* __restrict__ CBASE,
                             ORDT* __restrict__ ORD, int64_t N, int K,
                             int chunk, int nchunks) {
  pdl_wait();  // PDL: predecessor writes visible here
  extern __shared__ int cur[];  // K ints
  const int b = blockIdx.y;
  const int c0 = blockIdx.x;
  int32_t* base = CBASE + ((int64_t)b * nchunks + c0) * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) cur[i] = base[i];
  __syncthreads();
  const int64_t p0 = (int64_t)c0 * chunk;
  const int64_t pend = min(p0 + chunk, N);
  const int32_t* idb = IDS + (int64_t)b * N;
  ORDT* ordb = ORD + (int64_t)b * N;
  for (int64_t p = p0 + threadIdx.x; p < pend; p += blockDim.x) {
    const int label = idb[p];
    const int pos = atomicAdd(&cur[label], 1);
    ordb[pos] = (ORDT)p;
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// reduce: CTA = one (label, chunk-of-up-to-C-points) pair, found by binary
// search over CHUNK_OFF. Warps stride points; lane l accumulates dims
// {4l..4l+3} in registers, then one global reduction per dim per CTA into SC.
//
// Fused finalize: the per-label
// chunk budget DONEC[b,k] (plan-written) is decremented once per chunk CTA
// after its SC accumulation; when the counter observes 1 this CTA is the
// label's final chunk and performs the centroid/HC finalization inline,
// replacing the separate finalize launch. Fast path: a single-chunk label's
// sole CTA already holds the full row in registers — it finalizes straight
// from them and writes NO SC dims slot at all (the arena must stay all-zero
// for the next reduce pass). Multi-chunk labels accumulate into SC with
// atomics, and the last-arriver CTA re-reads the row from L2 under a
// release/acquire pair on DONEC, finalizes, and zeroes the dims back. Empty
// labels own no chunk and are finalized by plan_kernel. Unlike the sm103
// file, SC rows are 129-float strided with the count at slot DK (written by
// plan each iteration); the 516B stride is not float4-aligned, so the rare
// last-arriver re-read/zero uses scalar accesses.
#define REDUCE_C 512
// NW warps per block, kept at 8 (256 threads): a 4-warp configuration on the
// many-tiny-cluster regime showed no benefit, so a single 8-warp
// instantiation is dispatched for all shapes.
// minBlocks=8 pins this kernel to 32 registers/CTA-slot: the fused finalize
// tail must not cost gather-loop occupancy (32 -> 40 regs drops resident
// CTAs/SM from 8 to 6).
template <typename ORDT, int NW>
__global__ void __launch_bounds__(NW * 32, 8) reduce_kernel(
    const bf16* __restrict__ X, const ORDT* __restrict__ ORD,
    const int32_t* __restrict__ OFF, const int32_t* __restrict__ CHUNK_OFF,
    float* __restrict__ SC, int64_t N, int K,
    const int2* __restrict__ CIDMAP, int64_t rchunks,
    bf16* __restrict__ Cnew, float* __restrict__ HC,
    int32_t* __restrict__ DONEC) {
  pdl_wait();  // PDL: predecessor writes visible here
  const int b = blockIdx.y;
  const int cid = blockIdx.x;
  if (cid >= CHUNK_OFF[b * (K + 1) + K]) return;
  // plan_kernel writes cid -> (k, s0) directly; one 8B load replaces the
  // per-block binary search over CHUNK_OFF (log2(K) dependent L2 hits).
  const int2 km = CIDMAP[(int64_t)b * rchunks + cid];
  const int k = km.x;
  const int64_t s0 = (int64_t)km.y;
  const int64_t s1 = min(s0 + REDUCE_C, (int64_t)OFF[b * (K + 1) + k + 1]);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const bf16* xb = X + (int64_t)b * N * DK;
  const ORDT* ordb = ORD + (int64_t)b * N;
  __shared__ float red[NW][128];
  const int len = (int)(s1 - s0);
  const ORDT* ords = ordb + s0;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  // one 8B vector load per lane covers the whole 256B point row per warp:
  // lane l owns dims {4l, 4l+1, 4l+2, 4l+3}. The ord entry is read directly
  // from global (hot in L2 right after place_kernel); the UD-unrolled chains
  // hide the dependent LDG latency, so dropping the smem staging copy and its
  // block-wide sync removes per-block latency for the many small-cluster
  // blocks this kernel is dominated by.
  constexpr int UD = 16;  // unroll depth: independent gather chains in flight
  int s = warp;
  for (; s + (UD - 1) * NW < len; s += UD * NW) {
    int o[UD];
    uint2 p[UD];
#pragma unroll
    for (int u = 0; u < UD; ++u) o[u] = ords[s + u * NW];
#pragma unroll
    for (int u = 0; u < UD; ++u)
      p[u] = reinterpret_cast<const uint2*>(xb + (int64_t)o[u] * DK)[lane];
#pragma unroll
    for (int u = 0; u < UD; ++u) {
      const bf16* e = reinterpret_cast<const bf16*>(&p[u]);
      a0 += __bfloat162float(e[0]);
      a1 += __bfloat162float(e[1]);
      a2 += __bfloat162float(e[2]);
      a3 += __bfloat162float(e[3]);
    }
  }
  // mid-length chunks (the UD=16 loop needs >=121 points for warp 0, so e.g.
  // K=1000 rows with ~76-point clusters never enter it): UD2 independent
  // gather chains cover 25..120-point chunks before the scalar tail.
  constexpr int UD2 = 4;
  for (; s + (UD2 - 1) * NW < len; s += UD2 * NW) {
    int o[UD2];
    uint2 p[UD2];
#pragma unroll
    for (int u = 0; u < UD2; ++u) o[u] = ords[s + u * NW];
#pragma unroll
    for (int u = 0; u < UD2; ++u)
      p[u] = reinterpret_cast<const uint2*>(xb + (int64_t)o[u] * DK)[lane];
#pragma unroll
    for (int u = 0; u < UD2; ++u) {
      const bf16* e = reinterpret_cast<const bf16*>(&p[u]);
      a0 += __bfloat162float(e[0]);
      a1 += __bfloat162float(e[1]);
      a2 += __bfloat162float(e[2]);
      a3 += __bfloat162float(e[3]);
    }
  }
  for (; s < len; s += NW) {
    const uint2 pk = reinterpret_cast<const uint2*>(xb + (int64_t)ords[s] * DK)[lane];
    const bf16* e = reinterpret_cast<const bf16*>(&pk);
    a0 += __bfloat162float(e[0]);
    a1 += __bfloat162float(e[1]);
    a2 += __bfloat162float(e[2]);
    a3 += __bfloat162float(e[3]);
  }
  red[warp][4 * lane] = a0;
  red[warp][4 * lane + 1] = a1;
  red[warp][4 * lane + 2] = a2;
  red[warp][4 * lane + 3] = a3;
  __syncthreads();
  if (warp == 0) {
    // per-dim totals in registers, identical reduction order to the old
    // SC-store + finalize pair (w ascending), so fused single-chunk
    // finalization is bit-exact with the previous two-kernel sequence.
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * lane + j;
      float v = red[0][d];
#pragma unroll
      for (int w = 1; w < NW; ++w) v += red[w][d];
      s[j] = v;
    }
    // single-chunk clusters have exactly one writer block per SC row: this
    // CTA is trivially the label's final chunk, so it finalizes from
    // registers and leaves the SC dims slots untouched (still all-zero for
    // the next pass). Multi-chunk clusters (big skew) keep atomics.
    const bool single =
        (s0 == (int64_t)OFF[b * (K + 1) + k]) && (s1 == (int64_t)OFF[b * (K + 1) + k + 1]);
    float* dst = SC + ((int64_t)b * K + k) * 129;
    if (single) {
      const float count = dst[DK];  // plan-written this iteration
      bf16 nv[4];
      float ss = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        nv[j] = __float2bfloat16(s[j] / count);
        const float fv = __bfloat162float(nv[j]);
        ss += fv * fv;
      }
      reinterpret_cast<uint2*>(Cnew + ((int64_t)b * K + k) * DK)[lane] =
          *reinterpret_cast<const uint2*>(nv);
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) HC[(int64_t)b * K + k] = 0.5f * ss;
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) atomicAdd(dst + 4 * lane + j, s[j]);
      __threadfence();  // release: contributions visible before the tally
      int last = 0;
      // warp-uniform arrival order: one lane decrements the label's chunk
      // budget; prev == 1 means every other chunk already arrived.
      __syncwarp();
      if (lane == 0)
        last = (atomicAdd(&DONEC[(int64_t)b * K + k], -1) == 1);
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();  // acquire: pairs with the other chunks' release
        // re-read the summed row (L2-hot), finalize, and zero the dims back
        // so the arena is all-zero at the next reduce entry (zero_sc
        // semantics). Scalar accesses: the 129-float row stride is not
        // float4-aligned.
        const float count = dst[DK];
        bf16 nv[4];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float v = dst[4 * lane + j];
          nv[j] = __float2bfloat16(v / count);
          const float fv = __bfloat162float(nv[j]);
          ss += fv * fv;
        }
        reinterpret_cast<uint2*>(Cnew + ((int64_t)b * K + k) * DK)[lane] =
            *reinterpret_cast<const uint2*>(nv);
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[4 * lane + j] = 0.f;
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) HC[(int64_t)b * K + k] = 0.5f * ss;
      }
    }
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// direct scatter update for small N (launch-latency regime): warp per row,
// SC[row-label] += x row via 4 lane-strided atomics; SC[...,128] += 1.
// Replaces the hist/plan/place/reduce kernel chain (4 launches) when N is
// small enough that the atomic volume (N x 128) is trivial.
__global__ void __launch_bounds__(256) scatter_small_kernel(
    const bf16* __restrict__ X, const int32_t* __restrict__ IDS,
    float* __restrict__ SC, int64_t N, int K) {
  pdl_wait();  // PDL: predecessor writes visible here
  const int b = blockIdx.y;
  const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= N) return;
  const int lane = threadIdx.x & 31;
  const bf16* xr = X + ((int64_t)b * N + row) * DK;
  const int k = IDS[(int64_t)b * N + row];
  float* dst = SC + ((int64_t)b * K + k) * 129;
  const uint2 pk = reinterpret_cast<const uint2*>(xr)[lane];
  const bf16* e = reinterpret_cast<const bf16*>(&pk);
  atomicAdd(dst + 4 * lane + 0, __bfloat162float(e[0]));
  atomicAdd(dst + 4 * lane + 1, __bfloat162float(e[1]));
  atomicAdd(dst + 4 * lane + 2, __bfloat162float(e[2]));
  atomicAdd(dst + 4 * lane + 3, __bfloat162float(e[3]));
  if (lane == 0) atomicAdd(dst + DK, 1.f);
  pdl_fire();  // PDL: all global stores above are complete
}

// finalize: new = count>0 ? bf16(sum/count) : prev; hcn of the new row.
// block of (DK, 8) threads, 32 rows per block (4 rows per thread): the four
// independent load/store chains hide L2 latency.
// When zero_sc is set, also zero the sc row right after reading it: the
// buffer then starts the next iteration already zeroed, replacing the
// separate per-iteration memset kernel (finalize reads every slot anyway).
#define FIN_RPB 32
__global__ void finalize_kernel(float* __restrict__ SC,
                                const bf16* __restrict__ Cold,
                                bf16* __restrict__ Cnew, float* __restrict__ HC,
                                int64_t rows, int zero_sc) {
  pdl_wait();  // PDL: predecessor writes visible here
  const int d = threadIdx.x;
  const int y = threadIdx.y;  // 0..7
  const int64_t row0 = (int64_t)blockIdx.x * FIN_RPB;
  __shared__ float red[FIN_RPB][4];
  const int warp = d >> 5, lane = d & 31;
#pragma unroll
  for (int j = 0; j < FIN_RPB / 8; ++j) {
    const int rr = y + j * 8;
    const int64_t row = row0 + rr;
    float s = 0.f;
    if (row < rows) {
      float* sc = SC + row * 129;
      const float count = sc[DK];
      bf16 nv;
      if (count > 0.f) {
        nv = __float2bfloat16(sc[d] / count);
      } else {
        nv = Cold[row * DK + d];
      }
      Cnew[row * DK + d] = nv;
      s = __bfloat162float(nv);
      s = s * s;
      // dims: each thread zeroes only the slot it already read (race-free)
      if (zero_sc) sc[d] = 0.f;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red[rr][warp] = s;
  }
  __syncthreads();
  if (warp == 0 && y == 0) {
    const int64_t row = row0 + lane;
    if (row < rows) {
      // count slot: zeroed here, after the barrier — every other thread has
      // already read its count by now.
      if (zero_sc) SC[row * 129 + DK] = 0.f;
      HC[row] = 0.5f * (red[lane][0] + red[lane][1] + red[lane][2] + red[lane][3]);
    }
  }
  pdl_fire();  // PDL: all global stores above are complete
}

// (A single-launch zero+scatter+finalize fusion for the small-N regime was
// benchmarked and REVERTED: with one CTA per batch the scatter/finalize loops
// serialize (46-360us on N>=130 rows vs ~20-26us for the standalone
// multi-CTA kernels), so the separate scatter_small + finalize kernels and
// the sc.zero_() fill stay.)

static inline void check_inputs(const at::Tensor& x, const at::Tensor& c) {
  TORCH_CHECK(x.is_cuda() && c.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(c.scalar_type() == at::kBFloat16, "centroids must be bf16");
  TORCH_CHECK(x.dim() == 3 && x.size(2) == DK, "x must be [B,N,128]");
  TORCH_CHECK(c.dim() == 3 && c.size(0) == x.size(0) && c.size(2) == DK,
              "centroids must be [B,K,128]");
  TORCH_CHECK(x.is_contiguous() && c.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(x.size(1) >= 1, "N must be >= 1");
  TORCH_CHECK(c.size(1) >= 1 && c.size(1) <= 1024, "K out of range");
}

// PDL launch wrapper: attaches cudaLaunchAttributeProgrammaticStreamSerialization
// so the previous grid's pdl_fire() may release this grid early while its
// pdl_wait() still enforces full predecessor visibility. Unlike <<< >>>, the
// cudaLaunchKernelEx template marshals argument VALUES without conversion, so
// every call site must pass arguments whose types already match the kernel
// parameters (explicit (int64_t)/(T*)nullptr casts where needed).
template <typename Fn, typename... Args>
static inline void pdl_launch(Fn kernel, dim3 grid, dim3 block, size_t smem,
                              cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

static void run_assign(const at::Tensor& x, const at::Tensor& c, at::Tensor& hc,
                       at::Tensor& ids, cudaStream_t stream, bool need_hc,
                       float* sc_zero = nullptr) {
  const int B = x.size(0), N = x.size(1), K = c.size(1);
  const int64_t rows = (int64_t)B * K;
  // The small-N assign path (assign_kernel) computes hc in-kernel per chunk
  // (bitwise-identical to hcnorm_kernel), so the standalone launch runs only
  // for the UMMA path that still reads HC from global memory.
  if (need_hc && N >= 4096) {
    dim3 block(32, 16);  // 16 rows/block: 4x fewer blocks, kills launch-bound latency
    dim3 grid((unsigned)((rows + block.y - 1) / block.y));
    pdl_launch(hcnorm_kernel, grid, block, 0, stream,
               as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(), rows);
  }
  if (N >= 4096) {
    // sm100-only source (the extension loader builds kmeans_lloyd_sm103.cu
    // instead when capability >= (10,3)).
    constexpr size_t cp2xsmem = 65536 + 16384 * 2 + 4096 + 16 + 1024 + 1024 + 16;
    constexpr size_t cp2xdecsmem = cp2xsmem + 16;
    static bool cpattr_set = false;
    if (!cpattr_set) {
      cudaFuncSetAttribute(assign_umma2x_cp_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)cp2xsmem);
      cudaFuncSetAttribute(assign_umma2x_dec_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)cp2xdecsmem);
      cudaFuncSetAttribute(assign_umma2x_dec_k293_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)cp2xdecsmem);
      cudaFuncSetAttribute(assign_umma2x_dec_k300_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)cp2xdecsmem);
      cpattr_set = true;
    }
    dim3 ublock(256);
    dim3 ugrid((N + 255) / 256, B);
    // decoupled-MMA variant pays off only on filled grids (measured regression
    // on boundary_4096_k1024-class grids with tiles*B far below one wave):
    // gate at one full wave of 2-CTA/SM residency on GB200 (148 CTAs).
    if ((((N + 255) / 256) * B) >= 148) {
      if (K == 293) {  // longcat rows: branch-free 5-chunk specialization
        pdl_launch(assign_umma2x_dec_k293_kernel, ugrid, ublock, cp2xdecsmem,
                   stream,
                   as_bf16(x.data_ptr<at::BFloat16>()),
                   as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
                   ids.data_ptr<int32_t>(), N, K);
        return;
      }
      if (K == 300) {  // wan21 query row: branch-free 5-chunk specialization
        pdl_launch(assign_umma2x_dec_k300_kernel, ugrid, ublock, cp2xdecsmem,
                   stream,
                   as_bf16(x.data_ptr<at::BFloat16>()),
                   as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
                   ids.data_ptr<int32_t>(), N, K);
        return;
      }
      pdl_launch(assign_umma2x_dec_kernel, ugrid, ublock, cp2xdecsmem, stream,
                 as_bf16(x.data_ptr<at::BFloat16>()),
                 as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
                 ids.data_ptr<int32_t>(), N, K);
      return;
    }
    pdl_launch(assign_umma2x_cp_kernel, ugrid, ublock, cp2xsmem, stream,
               as_bf16(x.data_ptr<at::BFloat16>()),
               as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
               ids.data_ptr<int32_t>(), N, K);
    return;
  }
  const size_t smem = sizeof(uint32_t) * KC * (CPW / 2) + sizeof(float) * KC;
  dim3 block(THREADS);
  const int ntiles = (N + BM - 1) / BM;
  const int nch = (K + KC - 1) / KC;
  // split-K for small grids: spreading centroid chunks across CTAs fills the
  // machine when ntiles*B is far below the SM count (e.g. B=2, N<=2048, K~1k).
  int ksplits = 1;
  if (nch >= 8) {
    const int fill = ntiles * B;
    int want = (96 + fill - 1) / fill;
    ksplits = min(min(want, nch / 2), 8);
    if (ksplits < 1) ksplits = 1;
  }
  const int cps = (nch + ksplits - 1) / ksplits;
  ksplits = (nch + cps - 1) / cps;
  if (ksplits > 1) {
    auto opts_f = at::TensorOptions().device(x.device()).dtype(at::kFloat);
    auto opts_i = at::TensorOptions().device(x.device()).dtype(at::kInt);
    at::Tensor pv = at::empty({(int64_t)B * ksplits * N}, opts_f);
    at::Tensor pi = at::empty({(int64_t)B * ksplits * N}, opts_i);
    dim3 grid((unsigned)(ntiles * ksplits), B);
    pdl_launch(assign_kernel, grid, block, smem, stream,
        as_bf16(x.data_ptr<at::BFloat16>()), as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
        (int32_t*)nullptr, N, K, ntiles, ksplits, cps, pv.data_ptr<float>(),
        pi.data_ptr<int32_t>(), sc_zero);
    dim3 mgrid((N + 255) / 256, B);
    pdl_launch(merge_assign_kernel, mgrid, 256, 0, stream,
        pv.data_ptr<float>(), pi.data_ptr<int32_t>(), ids.data_ptr<int32_t>(),
        (int64_t)N, ksplits);
    return;
  }
  dim3 grid((unsigned)ntiles, B);
  pdl_launch(assign_kernel, grid, block, smem, stream,
      as_bf16(x.data_ptr<at::BFloat16>()), as_bf16(c.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
      ids.data_ptr<int32_t>(), N, K, ntiles, 1, nch, (float*)nullptr, (int32_t*)nullptr, sc_zero);
}

// One sorted-update pass over batch slices of `ub` rows (ub == B when
// unsliced). All update scratch (SC/CHIST/OFF/CHUNK_OFF/ORD/CIDMAP) is
// slice-sized and reused across slices; every kernel in the chain addresses
// batch rows independently (blockIdx.y/.x as local batch), so a sliced pass
// produces bitwise-identical results to the full-batch pass while reducing
// the simultaneous-scratch footprint by (B - ub)/B.
template <typename ORDT>
static void run_sorted_update(const bf16* __restrict__ xp, const int32_t* idsp,
                              bf16* curp, bf16* dstp, float* hcp, float* scp,
                              int32_t* chistp, int32_t* offp, int32_t* chunk_offp,
                              ORDT* ordp, int32_t* cidmapp, int32_t* donecp,
                              int64_t N, int K,
                              int B, int ub, int64_t rchunks,
                              cudaStream_t stream) {
  const int hchunk = 2048;
  const int nc = (int)((N + hchunk - 1) / hchunk);
  const size_t hsmem = sizeof(int) * (size_t)K;
  const int staged = (int)((int64_t)nc * K * 4 <= 200 * 1024);
  const size_t psmem = staged ? (size_t)nc * K * 4 : 0;
  static bool planattr_set = false;
  if (!planattr_set) {
    cudaFuncSetAttribute(plan_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         200 * 1024);
    planattr_set = true;
  }
  for (int b0 = 0; b0 < B; b0 += ub) {
    const int bs = min(ub, B - b0);
    dim3 hgrid((unsigned)nc, (unsigned)bs);
    const int32_t* idsb = idsp + (int64_t)b0 * N;
    bf16* cold = curp + (int64_t)b0 * K * DK;
    bf16* cnew = dstp + (int64_t)b0 * K * DK;
    float* hcs = hcp + (int64_t)b0 * K;
    pdl_launch(hist_kernel, hgrid, 256, hsmem, stream, idsb, chistp, N, K,
               hchunk, nc);
    pdl_launch(plan_kernel, bs, 1024, psmem, stream,
               chistp, offp, chunk_offp, scp, K, REDUCE_C, nc, staged,
               reinterpret_cast<int2*>(cidmapp), rchunks,
               (const bf16*)cold, cnew, hcs, donecp);
    pdl_launch(place_kernel<ORDT>, hgrid, 256, hsmem, stream, idsb, chistp,
               ordp, N, K, hchunk, nc);
    dim3 rgrid((unsigned)rchunks, (unsigned)bs);
    // 8-warp blocks (256 threads) are used across all shapes. The finalize
    // launch is fused into plan (empty labels) + reduce (populated labels) —
    // one fewer launch per iteration and no SC dims traffic for single-chunk
    // labels.
    pdl_launch(reduce_kernel<ORDT, 8>, rgrid, 256, 0, stream,
               xp + (int64_t)b0 * N * DK, ordp, offp, chunk_offp, scp, N, K,
               reinterpret_cast<const int2*>(cidmapp), rchunks, cnew, hcs,
               donecp);
  }
}

std::vector<at::Tensor> kda_lloyd(at::Tensor x, at::Tensor centroids,
                                  int64_t iterations) {
  check_inputs(x, centroids);
  const int B = x.size(0), N = x.size(1), K = centroids.size(1);
  auto opts_f = at::TensorOptions().device(x.device()).dtype(at::kFloat);
  auto opts_i = at::TensorOptions().device(x.device()).dtype(at::kInt);
  auto opts_b = at::TensorOptions().device(x.device()).dtype(at::kBFloat16);
  auto opts_u8 = at::TensorOptions().device(x.device()).dtype(at::kByte);

  at::Tensor ids = at::empty({B, N}, opts_i);
  at::Tensor hc = at::empty({B, K}, opts_f);
  const bool predict = (iterations <= 0);
  const int iters = predict ? 1 : (int)iterations;
  // centroid working set: multi-iteration runs own a single clone and
  // finalize updates it in place (safe: Cold[row*DK+d] is read and
  // Cnew[row*DK+d] written by the same thread only), so no second B*K*128
  // bf16 buffer is ever alive. Single-iteration calls keep one fresh output
  // buffer and never write the caller's input centroids.
  at::Tensor cur = (iters <= 1) ? centroids : centroids.clone();
  at::Tensor out;  // iters==1 (train) only: the single fresh output buffer
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor sc, chist, off, chunk_off, ord, cidmap, donec;
  const int64_t rows = (int64_t)B * K;
  const bool small_update = (N <= 4096);
  // ord entries are global point ids in [0,N): a uint16 table halves the
  // B*N placement buffer whenever N fits (identical values, narrower loads).
  const bool ord16 = (N <= 65535);
  // batch-sliced centroid update for N > 64K: the sorted-update scratch
  // (ord ~ B*N, SC ~ B*K*129) is the dominant peak-memory term there, so
  // update in batch slices of ceil(2B/3), reusing one slice-sized scratch set.
  int ub = B;
  if (!predict && !small_update && !ord16) ub = std::max(8, (2 * B) / 3);
  if (ub > B) ub = B;
  int64_t rchunks = 0;
  if (!predict) {
    if (iters == 1) out = at::empty({B, K, DK}, opts_b);
    sc = at::empty({(int64_t)ub * K * 129}, opts_f);
    // zeroed once here; the update chain keeps the arena all-zero at every
    // boundary: the small-N path's finalize re-zeroes the rows it read, and
    // the sorted path's fused reduce finalizes single-chunk labels without
    // touching SC and zeroes multi-chunk rows back inline. On the small-N
    // assign path the first iteration's assign_kernel zeroes the SC slices
    // instead (tile0/split0 CTA per batch, overlapped with the other CTAs'
    // compute; stream-ordered before the scatter), so the standalone fill
    // launch is dropped there.
    const bool assign_zeroes = small_update && N < 4096 && !predict;
    if (!assign_zeroes) sc.zero_();
    if (!small_update) {
      // 2048-point chunks: hist/place get ~4x more CTAs than 8192, filling
      // the machine at large N.
      const int hchunk = 2048;
      const int nc = (int)((N + hchunk - 1) / hchunk);
      // one per-chunk buffer: plan_kernel turns the histogram into the
      // placement cursors in place (read-before-write; see its comment)
      chist = at::empty({(int64_t)ub * nc * K}, opts_i);
      off = at::empty({(int64_t)ub * (K + 1)}, opts_i);
      chunk_off = at::empty({(int64_t)ub * (K + 1)}, opts_i);
      ord = ord16 ? at::empty({(int64_t)ub * N * 2}, opts_u8)
                  : at::empty({(int64_t)ub * N}, opts_i);
      rchunks = K + (int64_t)((N + REDUCE_C - 1) / REDUCE_C);
      cidmap = at::empty({(int64_t)ub * rchunks * 2}, opts_i);
      // per-label reduce-chunk budget for the fused finalize (plan rewrites
      // it every iteration; no zeroing needed)
      donec = at::empty({(int64_t)ub * K}, opts_i);
    }
  }

  for (int it = 0; it < iters; ++it) {
    // Recompute hc every iteration in run_assign: hcnorm's summation order
    // matches the reference more closely than finalize's (measured label
    // agreement 0.9958 -> 0.9948 regression on the i10 row when reusing
    // finalize's hc), which costs more than the ~4us/iter saved.
    run_assign(x, cur, hc, ids, stream, /*need_hc=*/true,
               (!predict && small_update && N < 4096 && it == 0)
                   ? sc.data_ptr<float>()
                   : nullptr);
    if (predict) break;
    // finalize destination: the multi-iteration clone is updated in place;
    // the iters==1 path writes its single fresh output buffer instead, so
    // the caller's input centroids are never modified.
    at::Tensor& dst = (iters == 1) ? out : cur;
    if (small_update) {
      dim3 sgrid((unsigned)((N + 7) / 8), B);
      pdl_launch(scatter_small_kernel, sgrid, 256, 0, stream,
                 as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
                 sc.data_ptr<float>(), (int64_t)N, K);
      pdl_launch(finalize_kernel,
                 (unsigned)((rows + FIN_RPB - 1) / FIN_RPB), dim3(DK, 8), 0,
                 stream,
                 sc.data_ptr<float>(), as_bf16(cur.data_ptr<at::BFloat16>()),
                 as_bf16(dst.data_ptr<at::BFloat16>()),
                 hc.data_ptr<float>(), rows, /*zero_sc=*/1);
    } else {
      if (ord16) {
        run_sorted_update<uint16_t>(
            as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
            as_bf16(cur.data_ptr<at::BFloat16>()),
            as_bf16(dst.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
            sc.data_ptr<float>(), chist.data_ptr<int32_t>(),
            off.data_ptr<int32_t>(), chunk_off.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(ord.data_ptr()),
            cidmap.data_ptr<int32_t>(), donec.data_ptr<int32_t>(), N, K, B,
            ub, rchunks, stream);
      } else {
        run_sorted_update<int32_t>(
            as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
            as_bf16(cur.data_ptr<at::BFloat16>()),
            as_bf16(dst.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
            sc.data_ptr<float>(), chist.data_ptr<int32_t>(),
            off.data_ptr<int32_t>(), chunk_off.data_ptr<int32_t>(),
            ord.data_ptr<int32_t>(), cidmap.data_ptr<int32_t>(),
            donec.data_ptr<int32_t>(), N, K, B, ub, rchunks, stream);
      }
    }
    // no buffer swap: multi-iteration updates landed in place on cur; the
    // iters==1 path wrote out and the loop exits after this pass
  }
  return {ids, (!predict && iters == 1) ? out : cur};
}

void kda_lloyd_into(at::Tensor x, at::Tensor centroids, int64_t iterations,
                    at::Tensor out_ids, at::Tensor out_cents) {
  check_inputs(x, centroids);
  const int B = x.size(0), N = x.size(1), K = centroids.size(1);
  auto opts_f = at::TensorOptions().device(x.device()).dtype(at::kFloat);
  auto opts_i = at::TensorOptions().device(x.device()).dtype(at::kInt);
  auto opts_b = at::TensorOptions().device(x.device()).dtype(at::kBFloat16);
  auto opts_u8 = at::TensorOptions().device(x.device()).dtype(at::kByte);

  at::Tensor& ids = out_ids;
  at::Tensor hc = at::empty({B, K}, opts_f);
  const bool predict = (iterations <= 0);
  const int iters = predict ? 1 : (int)iterations;
  TORCH_CHECK(out_ids.is_cuda() && out_ids.dim() == 2 &&
                  out_ids.size(0) == B && out_ids.size(1) == N &&
                  out_ids.scalar_type() == at::kInt && out_ids.is_contiguous(),
              "kda_lloyd_into: out_ids must be a contiguous CUDA int32 [B,N]");
  if (!predict) {
    TORCH_CHECK(out_cents.is_cuda() && out_cents.dim() == 3 &&
                    out_cents.size(0) == B && out_cents.size(1) == K &&
                    out_cents.size(2) == x.size(2) &&
                    out_cents.scalar_type() == at::kBFloat16 &&
                    out_cents.is_contiguous(),
                "kda_lloyd_into: out_cents must be a contiguous CUDA bf16 [B,K,128]");
  }
  // centroid working set: multi-iteration runs own a single clone and
  // finalize updates it in place (safe: Cold[row*DK+d] is read and
  // Cnew[row*DK+d] written by the same thread only), so no second B*K*128
  // bf16 buffer is ever alive. Single-iteration calls keep one fresh output
  // buffer and never write the caller's input centroids.
  at::Tensor cur = (iters <= 1) ? centroids : centroids.clone();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor sc, chist, off, chunk_off, ord, cidmap, donec;
  const int64_t rows = (int64_t)B * K;
  const bool small_update = (N <= 4096);
  // ord entries are global point ids in [0,N): a uint16 table halves the
  // B*N placement buffer whenever N fits (identical values, narrower loads).
  const bool ord16 = (N <= 65535);
  // batch-sliced centroid update for N > 64K: the sorted-update scratch
  // (ord ~ B*N, SC ~ B*K*129) is the dominant peak-memory term there, so
  // update in batch slices of ceil(2B/3), reusing one slice-sized scratch set.
  int ub = B;
  if (!predict && !small_update && !ord16) ub = std::max(8, (2 * B) / 3);
  if (ub > B) ub = B;
  int64_t rchunks = 0;
  if (!predict) {
    sc = at::empty({(int64_t)ub * K * 129}, opts_f);
    // zeroed once here; the update chain keeps the arena all-zero at every
    // boundary: the small-N path's finalize re-zeroes the rows it read, and
    // the sorted path's fused reduce finalizes single-chunk labels without
    // touching SC and zeroes multi-chunk rows back inline. On the small-N
    // assign path the first iteration's assign_kernel zeroes the SC slices
    // instead (tile0/split0 CTA per batch, overlapped with the other CTAs'
    // compute; stream-ordered before the scatter), so the standalone fill
    // launch is dropped there.
    const bool assign_zeroes = small_update && N < 4096 && !predict;
    if (!assign_zeroes) sc.zero_();
    if (!small_update) {
      // 2048-point chunks: hist/place get ~4x more CTAs than 8192, filling
      // the machine at large N.
      const int hchunk = 2048;
      const int nc = (int)((N + hchunk - 1) / hchunk);
      // one per-chunk buffer: plan_kernel turns the histogram into the
      // placement cursors in place (read-before-write; see its comment)
      chist = at::empty({(int64_t)ub * nc * K}, opts_i);
      off = at::empty({(int64_t)ub * (K + 1)}, opts_i);
      chunk_off = at::empty({(int64_t)ub * (K + 1)}, opts_i);
      ord = ord16 ? at::empty({(int64_t)ub * N * 2}, opts_u8)
                  : at::empty({(int64_t)ub * N}, opts_i);
      rchunks = K + (int64_t)((N + REDUCE_C - 1) / REDUCE_C);
      cidmap = at::empty({(int64_t)ub * rchunks * 2}, opts_i);
      // per-label reduce-chunk budget for the fused finalize (plan rewrites
      // it every iteration; no zeroing needed)
      donec = at::empty({(int64_t)ub * K}, opts_i);
    }
  }

  for (int it = 0; it < iters; ++it) {
    // Recompute hc every iteration in run_assign: hcnorm's summation order
    // matches the reference more closely than finalize's (measured label
    // agreement 0.9958 -> 0.9948 regression on the i10 row when reusing
    // finalize's hc), which costs more than the ~4us/iter saved.
    run_assign(x, cur, hc, ids, stream, /*need_hc=*/true,
               (!predict && small_update && N < 4096 && it == 0)
                   ? sc.data_ptr<float>()
                   : nullptr);
    if (predict) break;
    // finalize destination: the multi-iteration clone is updated in place;
    // the iters==1 path writes its single fresh output buffer instead, so
    // the caller's input centroids are never modified.
    at::Tensor& dst = (it == iters - 1) ? out_cents : cur;
    if (small_update) {
      dim3 sgrid((unsigned)((N + 7) / 8), B);
      pdl_launch(scatter_small_kernel, sgrid, 256, 0, stream,
                 as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
                 sc.data_ptr<float>(), (int64_t)N, K);
      pdl_launch(finalize_kernel,
                 (unsigned)((rows + FIN_RPB - 1) / FIN_RPB), dim3(DK, 8), 0,
                 stream,
                 sc.data_ptr<float>(), as_bf16(cur.data_ptr<at::BFloat16>()),
                 as_bf16(dst.data_ptr<at::BFloat16>()),
                 hc.data_ptr<float>(), rows, /*zero_sc=*/1);
    } else {
      if (ord16) {
        run_sorted_update<uint16_t>(
            as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
            as_bf16(cur.data_ptr<at::BFloat16>()),
            as_bf16(dst.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
            sc.data_ptr<float>(), chist.data_ptr<int32_t>(),
            off.data_ptr<int32_t>(), chunk_off.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(ord.data_ptr()),
            cidmap.data_ptr<int32_t>(), donec.data_ptr<int32_t>(), N, K, B,
            ub, rchunks, stream);
      } else {
        run_sorted_update<int32_t>(
            as_bf16(x.data_ptr<at::BFloat16>()), ids.data_ptr<int32_t>(),
            as_bf16(cur.data_ptr<at::BFloat16>()),
            as_bf16(dst.data_ptr<at::BFloat16>()), hc.data_ptr<float>(),
            sc.data_ptr<float>(), chist.data_ptr<int32_t>(),
            off.data_ptr<int32_t>(), chunk_off.data_ptr<int32_t>(),
            ord.data_ptr<int32_t>(), cidmap.data_ptr<int32_t>(),
            donec.data_ptr<int32_t>(), N, K, B, ub, rchunks, stream);
      }
    }
    // no buffer swap: multi-iteration updates landed in place on cur; the
    // iters==1 path wrote out_cents and the loop exits after this pass
  }
}
