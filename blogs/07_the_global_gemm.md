# The Global GEMM — Putting It All Together

**Difficulty:** Intermediate–Advanced
**Prerequisites:** [Tutorial 02: Partitioning](02_the_art_of_slicing.md), [Tutorial 04: TiledCopy](04_the_parallel_copy.md), [Tutorial 05: Swizzling](05_swizzling.md), [Tutorial 06: Hello, MMA](06_hello_mma.md)

## 1. The Problem (The "Why")

Tutorial 06 triggered a single Tensor Core instruction on a 16×8×16 (M×N×K) micro-tile. That's 2048 multiply-adds — impressive for one instruction, but real-world matrices are thousands of elements wide. You need to compute C(M×N) = A(M×K) × Bᵀ(N×K) where M, N, and K can each be 4096 or more.

The answer isn't one giant instruction. It's **tiling** — the same idea from Tutorial 02, now applied at every level:

| Level | What's tiled | Controlled by |
| :--- | :--- | :--- |
| **CTA grid** | Output C is divided into BLK_M×BLK_N tiles, one per thread block | `local_tile` + `blockIdx` |
| **K-loop** | The reduction dimension K is divided into BLK_K chunks, processed sequentially | The mainloop `for` loop |
| **Warp MMA** | Each BLK_M×BLK_N tile is further divided by the `TiledMMA` across 4 warps | `partition_A/B/C` + `gemm()` |

This is the structure of every high-performance GEMM on GPUs — from CUTLASS to cuBLAS. The names change but the three-level tiling is always the same. This tutorial writes that kernel from scratch, using nothing but the CuTe primitives you already know.

> **B200 Note:** On Hopper/Blackwell, the structure is identical but the data-movement layer changes: TiledCopy with `cp.async` or TMA replaces explicit loads, and the K-loop becomes a software pipeline (Tutorial 10). The compute layer (`TiledMMA` + `gemm()`) stays the same.

## 2. The Mental Model (The Visual)

### Running the Factory for a Full Production Day

Tutorials 03–06 built the factory:
- **Loading dock** (TiledCopy) — moves raw materials from the truck (global memory) into the warehouse (shared memory), using dollies (vectorized loads) and staggered shelving (swizzle).
- **Stamping press** (TiledMMA) — processes materials from the warehouse shelves into finished products (C accumulator).

Now we run the factory for a full **production day**:

```text
                    Global Memory (The Truck)
    ┌────────────────────────────────────────────────┐
    │  A (M × K)                  B (N × K)          │
    └──────────┬─────────────────────┬───────────────┘
               │                     │
    ═══════════╪═════════════════════╪══════ K-loop (the shift) ════
    ║          ▼                     ▼                             ║
    ║   ┌─────────────┐      ┌─────────────┐                      ║
    ║   │ A tile      │      │ B tile      │  ← TiledCopy         ║
    ║   │ (128 × 32)  │      │ (128 × 32)  │    gmem → smem       ║
    ║   └──────┬──────┘      └──────┬──────┘                      ║
    ║          │    Shared Memory    │                              ║
    ║          ▼    (swizzled)       ▼                              ║
    ║   ┌─────────────────────────────────┐                        ║
    ║   │         STAMPING PRESS          │  ← TiledMMA            ║
    ║   │    partition → fragment → gemm  │    smem → regs → MMA   ║
    ║   └────────────────┬────────────────┘                        ║
    ║                    │                                          ║
    ║              accum += partial C                               ║
    ║                                                              ║
    ═══════════════════════════════════════ repeat for each K tile ══

                    │
                    ▼  Epilogue (end of shift — ship the product)
             ┌──────────┐
             │  C tile   │  ← copy accum → gmem
             │ (128×128) │
             └──────────┘
```

Each thread block is an independent factory. The grid of blocks tiles the M×N output:

```text
Grid of CTAs over the output C (M × N):

         N →
    ┌─────────┬─────────┬─────────┬───┐
    │ CTA     │ CTA     │ CTA     │   │
  M │ (0,0)   │ (0,1)   │ (0,2)   │...│  Each CTA computes
  ↓ │ 128×128 │ 128×128 │ 128×128 │   │  one 128×128 tile of C
    ├─────────┼─────────┼─────────┼───┤
    │ CTA     │ CTA     │ CTA     │   │
    │ (1,0)   │ (1,1)   │ (1,2)   │...│
    │ 128×128 │ 128×128 │ 128×128 │   │
    ├─────────┼─────────┼─────────┼───┤
    │  ...    │  ...    │  ...    │   │
    └─────────┴─────────┴─────────┴───┘

grid = (M / 128, N / 128)
```

### The Three Phases

```text
Phase 1: COPY (the loading dock)
  gmem A[BLK_M × BLK_K] ──TiledCopy──▶ smem_A (swizzled)
  gmem B[BLK_N × BLK_K] ──TiledCopy──▶ smem_B (swizzled)
  __syncthreads()

Phase 2: COMPUTE (the stamping press)
  smem_A ──partition_A──▶ registers ──┐
                                      ├──gemm()──▶ accum += partial
  smem_B ──partition_B──▶ registers ──┘
  __syncthreads()

Phase 3: EPILOGUE (shipping)
  accum ──partition_C──▶ gmem C[BLK_M × BLK_N]
```

Phases 1 and 2 repeat `ceil(K / BLK_K)` times. Phase 3 happens once.

## 3. The Solution (The Code)

A complete GEMM kernel: `C (M×N, float) += A (M×K, half) × B^T (N×K, half)`. Uses 128 threads (4 warps), 128×128 CTA tiles, and SM80 Tensor Cores.

**Constraints:** M and N must be multiples of 128; K must be a multiple of 32. (Removing these constraints requires boundary masking — important for production, orthogonal to the core GEMM structure.)

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/swizzle.hpp>

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace cute;

// Tile sizes
constexpr int BLK_M = 128;
constexpr int BLK_N = 128;
constexpr int BLK_K = 32;
constexpr int NUM_THREADS = 128;  // 4 warps

__global__ void gemm_kernel(
    half const* __restrict__ A_ptr,
    half const* __restrict__ B_ptr,
    float*      __restrict__ C_ptr,
    int M, int N, int K)
{
    // SETUP: Layouts, TiledCopy, TiledMMA

    // Global memory tensors
    // A: (M, K) with K stride-1   B: (N, K) with K stride-1
    // C: (M, N) with N stride-1
    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    // CTA tiling: each block claims its tile of A, B, C
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);

    // gA: (BLK_M, BLK_K, k_tiles) — this block's strip of A, sliced along K
    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                          make_coord(get<0>(cta_coord), _));
    // gB: (BLK_N, BLK_K, k_tiles)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                          make_coord(get<1>(cta_coord), _));
    // gC: (BLK_M, BLK_N) — this block's output tile
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), cta_coord);

    // Shared memory layouts: K stride-1, swizzled
    // Swizzle<3,3,3>: M=3 → 8 halfs (128 bits) stay contiguous for vectorized loads
    auto sA_layout = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                    make_stride(Int<BLK_K>{}, Int<1>{})));
    auto sB_layout = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                    make_stride(Int<BLK_K>{}, Int<1>{})));

    // Shared memory allocation (static)
    __shared__ half smem_A[cosize_v<decltype(sA_layout)>];
    __shared__ half smem_B[cosize_v<decltype(sB_layout)>];

    auto sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

    // TiledCopy: gmem → smem via cp.async
    // Each atom loads 128 bits = 8 halfs along the K (stride-1) direction.
    // 128 threads, one per M-row. Each thread loads 32 K-values (4 × 128-bit async copies).
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half>{},
        Layout<Shape<_128, _1>>{},    // 128 threads all in M
        Layout<Shape<_1, _32>>{}      // 32 K-values per thread (4 atoms of 8 halfs)
    );
    // Coverage: (128 × 1, 1 × 32) = (128, 32) ✓

    // TiledMMA: the Tensor Core plan
    // SM80_16x8x16 atom, 4 warps arranged (2,2) in M×N.
    // Base coverage: (2×16, 2×8) = (32, 16) per atom group.
    // For 128×128: gemm() iterates 4× in M, 8× in N, 2× in K automatically.
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<_2, _2, _1>>{}   // 4 warps: 2 in M, 2 in N
    );

    // PARTITIONS: thread-level views

    // Copy partitions
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);   // (CPY_VEC, CPY_M, CPY_K, k_tiles)
    auto tAsA = thr_copy.partition_D(sA);   // (CPY_VEC, CPY_M, CPY_K)
    auto tBgB = thr_copy.partition_S(gB);   // same structure for B
    auto tBsB = thr_copy.partition_D(sB);

    // MMA partitions
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);    // (MMA_VAL, MMA_M, MMA_K)
    auto tCsB = thr_mma.partition_B(sB);    // (MMA_VAL, MMA_N, MMA_K)
    auto tCgC = thr_mma.partition_C(gC);    // (MMA_VAL, MMA_M, MMA_N)

    // Register fragments
    auto tCrA  = make_fragment_like(tCsA);
    auto tCrB  = make_fragment_like(tCsB);
    auto accum = make_fragment_like(tCgC);  // C accumulator (float)
    clear(accum);

    // MAINLOOP: iterate over K tiles

    int k_tiles = size<2>(gA);   // = K / BLK_K

    for (int k = 0; k < k_tiles; ++k) {

        // Phase 1: COPY gmem → smem
        copy(tiled_copy, tAgA(_, _, _, k), tAsA);
        copy(tiled_copy, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Phase 2: COMPUTE smem → registers → MMA
        copy(tCsA, tCrA);
        copy(tCsB, tCrB);
        gemm(tiled_mma, tCrA, tCrB, accum);

        __syncthreads();
    }

    // EPILOGUE: write C accumulator to global memory

    copy(accum, tCgC);
}

// Host: verification + timing
int main()
{
    constexpr int M = 512, N = 512, K = 256;

    // Allocate and initialize
    int sizeA = M * K, sizeB = N * K, sizeC = M * N;

    half *h_A = new half[sizeA];
    half *h_B = new half[sizeB];
    float *h_C = new float[sizeC];

    // A = all 1.0h, B = all 1.0h → C[m][n] = K
    for (int i = 0; i < sizeA; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < sizeB; ++i) h_B[i] = __float2half(1.0f);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(half));
    cudaMalloc(&d_B, sizeB * sizeof(half));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMemcpy(d_A, h_A, sizeA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(half), cudaMemcpyHostToDevice);

    // Launch
    dim3 grid(M / BLK_M, N / BLK_N);   // (4, 4) for 512×512
    dim3 block(NUM_THREADS);             // 128

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Verify
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    float expected = float(K);  // K * 1.0 * 1.0
    for (int i = 0; i < sizeC; ++i)
        max_err = fmaxf(max_err, fabsf(h_C[i] - expected));

    printf("GEMM: C(%d x %d) = A(%d x %d) * B^T(%d x %d)\n", M, N, M, K, N, K);
    printf("Tile: %d x %d x %d, Threads: %d\n\n", BLK_M, BLK_N, BLK_K, NUM_THREADS);

    // Print a 4×4 corner
    printf("C[0:4][0:4] (expected %.0f everywhere):\n", expected);
    for (int m = 0; m < 4; ++m) {
        printf("  ");
        for (int n = 0; n < 4; ++n)
            printf("%8.1f", h_C[m * N + n]);
        printf("\n");
    }
    printf("\nMax absolute error: %e\n", max_err);
    printf("Result: %s\n", max_err < 1e-3f ? "PASS" : "FAIL");

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 5, iters = 20;

    for (int i = 0; i < warmup; ++i)
        gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("\nPerformance: %.3f ms  (%.2f TFLOPS)\n", avg_ms, tflops);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
```

**Expected Output:**

```text
GEMM: C(512 x 512) = A(512 x 256) * B^T(512 x 256)
Tile: 128 x 128 x 32, Threads: 128

C[0:4][0:4] (expected 256 everywhere):
       256.0   256.0   256.0   256.0
       256.0   256.0   256.0   256.0
       256.0   256.0   256.0   256.0
       256.0   256.0   256.0   256.0

Max absolute error: 0.000000e+00
Result: PASS

Performance: X.XXX ms  (Y.YY TFLOPS)
```

Each entry of C is 256.0: the dot product of a length-256 all-ones row from A with a length-256 all-ones row from B. The K-loop ran `256 / 32 = 8` iterations, each accumulating a BLK_K=32 partial product. After 8 iterations, the accumulator held the full result.

## 4. Step-by-Step Explanation

### Setup Phase

**Lines: Global tensor creation with `make_tensor`**

```cpp
auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
```

Creates a CuTe tensor wrapping global memory. The stride `(K, 1)` means K is stride-1 — the TN layout required by our MMA atom (Tutorial 06). Note that `K` is a runtime value while `Int<1>{}` is static. CuTe handles mixed static/dynamic strides.

**Lines: CTA tiling with `local_tile`**

```cpp
auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                      make_coord(get<0>(cta_coord), _));
```

This is Tutorial 02's `local_tile` at the CTA level — the casino floor manager assigning tables:
- Divides A's M dimension into chunks of BLK_M (128). Block `blockIdx.x` claims its chunk.
- Divides A's K dimension into chunks of BLK_K (32). The underscore `_` means "give me all K-tiles as a third mode."
- Result shape: `(128, 32, k_tiles)` — this block's full strip of A, with `k_tiles = K / BLK_K` slices along K.

For C, both coordinates are fixed (no underscore):
```cpp
auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), cta_coord);
```
Result shape: `(128, 128)` — this block's one output tile.

**Lines: Shared memory with swizzle**

```cpp
auto sA_layout = composition(Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));
```

The base layout is `(128, 32):(32, 1)` — K stride-1, same as gmem. `composition(Swizzle<3,3,3>{}, ...)` applies the XOR remapping from Tutorial 05. The parameters: M=3 preserves 8-half (128-bit) contiguity for vectorized stores, B=3 scrambles 8-bank groups, S=3 separates the bit fields. This is the standard swizzle for half-precision data.

**Lines: TiledCopy construction**

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half>{},
    Layout<Shape<_128, _1>>{},
    Layout<Shape<_1, _32>>{}
);
```

This is Tutorial 04's supervisor's clipboard, now sized for the GEMM tile:
- **Copy_Atom:** `SM80_CP_ASYNC_CACHEALWAYS` — hardware async copy, 128 bits = 8 halfs per transaction along K (stride-1). Data goes directly from gmem to smem, bypassing registers.
- **thr_layout `(128, 1)`:** All 128 threads assigned to the M dimension — one thread per row.
- **val_layout `(1, 32)`:** Each thread copies 32 K-values (4 atoms × 8 halfs each).
- **Coverage:** `(128 × 1, 1 × 32) = (128, 32)` — exactly one smem tile. Each K-loop iteration fills the tile with one `copy()` call per matrix.

**Lines: TiledMMA construction**

```cpp
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
    Layout<Shape<_2, _2, _1>>{}
);
```

Tutorial 06's stamping press, now with multiple warps. The second argument arranges 4 warps in a 2×2 grid over M and N:
- Base atom covers 16M × 8N × 16K.
- 2×2 warps → combined base: 32M × 16N × 16K.
- For the 128×128×32 smem tile, `gemm()` internally loops 4× in M (128/32), 8× in N (128/16), and 2× in K (32/16). You don't write these loops — `gemm()` handles them.

### Partition Phase

**Lines: Copy partitions**

```cpp
auto tAgA = thr_copy.partition_S(gA);   // (CPY_VEC, CPY_M, CPY_K, k_tiles)
auto tAsA = thr_copy.partition_D(sA);   // (CPY_VEC, CPY_M, CPY_K)
```

Same as Tutorial 04: `partition_S` creates this thread's view of the source (gmem), `partition_D` for the destination (smem). The extra `k_tiles` mode on `tAgA` comes from gA's third mode — it passes through untouched. In the mainloop, `tAgA(_, _, _, k)` selects the k-th K-tile for copying.

**Lines: MMA partitions and fragments**

```cpp
auto tCsA = thr_mma.partition_A(sA);    // (MMA_VAL, MMA_M, MMA_K)
auto tCrA = make_fragment_like(tCsA);   // register copy
auto accum = make_fragment_like(tCgC);  // C accumulator
```

Same as Tutorial 06: `partition_A` maps smem elements to this thread's MMA fragment slots. `make_fragment_like` creates a register tensor with the same shape. The accumulator `accum` lives in registers for the entire kernel — it's never written to smem.

### Mainloop

```cpp
for (int k = 0; k < k_tiles; ++k) {
    copy(tiled_copy, tAgA(_, _, _, k), tAsA);    // gmem → smem
    copy(tiled_copy, tBgB(_, _, _, k), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(tCsA, tCrA);                            // smem → registers
    copy(tCsB, tCrB);
    gemm(tiled_mma, tCrA, tCrB, accum);          // MMA!
    __syncthreads();
}
```

Each iteration processes one BLK_K=32 slice of the K dimension:

1. **Copy** — TiledCopy loads A's k-th tile `(128×32)` and B's k-th tile `(128×32)` from global to shared memory. The swizzle is transparent — the TiledCopy writes through swizzled addresses automatically.

2. **Sync** — All 128 threads must finish writing to smem before any thread starts reading for the MMA.

3. **Compute** — Copy from swizzled smem to register fragments, then `gemm()` executes the MMA atoms. Because the tile is larger than one atom, `gemm()` internally loops over M, N, and K repetitions. The result **accumulates** into `accum` — each K-iteration adds to the running sum, computing the partial dot products.

4. **Sync** — All threads must finish reading smem before the next iteration overwrites it.

After `k_tiles` iterations, `accum` holds the complete C tile.

### Epilogue

```cpp
copy(accum, tCgC);
```

Each thread writes its accumulator fragment (128 floats for this config) to global memory via the MMA's C partition. As noted in Tutorial 06, the C fragments are scattered — each thread writes to non-contiguous addresses. For a production kernel, CUTLASS routes the epilogue through shared memory for coalesced writes. For correctness, the simple scattered write works fine.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy: The Full Production Day.**
- **CTA grid** = multiple independent factories, one per output tile. They never interact.
- **K-loop (mainloop)** = one production shift. Each iteration loads a batch of raw materials (A/B K-tiles via TiledCopy), processes them on the stamping press (TiledMMA + gemm), and adds the result to the running accumulator.
- **Epilogue** = end of shift. Ship the finished product (accumulated C tile) to global memory.

The three-level tiling is the single most important structural insight for GPU GEMM. Everything else — pipelining, double buffering, warpgroup MMA — is optimization on top of this skeleton.

**Tile size cheat sheet:**

| Parameter | Typical Values | Trade-off |
| :--- | :--- | :--- |
| BLK_M × BLK_N | 128×128, 128×256, 256×128 | Larger tiles → more compute per gmem load (better arithmetic intensity). But more smem usage → fewer CTAs per SM → less occupancy. |
| BLK_K | 32, 64 | Larger K → fewer mainloop iterations → less loop overhead. But more smem per tile (BLK_M × BLK_K × sizeof(half) per matrix). |
| Threads | 128 (4 warps), 256 (8 warps) | More threads → more copy bandwidth, but more MMA rep sharing overhead. |

**Why the swizzle parameters are `<3, 3, 3>` for half:**

- M=3: 2^3 = 8 halfs = 128 bits stay contiguous. Matches the 128-bit vectorized copy.
- B=3: 2^3 = 8 banks scrambled. Breaks the column-stride conflict pattern.
- S=3: Source bits `[6:9)`, target bits `[3:6)`. Non-overlapping.

Compare with Tutorial 05's `Swizzle<3, 2, 3>` for float: M=2 (4 floats = 128 bits). The only difference is the free-bit count M, which must match the element size for 128-bit vectorization.

**Why two `__syncthreads()` in the mainloop:**

```text
Iteration k:                    Iteration k+1:
  copy gmem → smem                copy gmem → smem ← OVERWRITES smem!
  ──sync── (1)                    ──sync──
  gemm reads smem                 gemm reads smem
  ──sync── (2)                    ──sync──
```

Sync (1): Ensure all threads finished writing smem before any thread reads it for gemm.
Sync (2): Ensure all threads finished reading smem before the next iteration's copy overwrites it.

Without sync (2), a fast thread could start writing the (k+1)-th A tile into smem while a slow thread is still reading the k-th A tile for its gemm.

**The epilogue bottleneck:**

Our epilogue uses `copy(accum, tCgC)` — each thread writes its C fragments directly to global memory with scattered stores. On A100, this wastes memory bus bandwidth because each 4-byte float write fetches a full 128-byte cache line.

CUTLASS's production epilogues:
1. Each warp writes its C fragment to smem (in a contiguous layout).
2. Threads read back from smem in a coalesced pattern and write to gmem.
3. Optional: fuse element-wise operations (bias add, ReLU) into the epilogue.

For this tutorial, the simple epilogue is correct and clear. Optimizing it is a straightforward extension.

**Performance expectations:**

This kernel will not match cuBLAS. Missing optimizations:
- No double-buffering (the copy and compute phases are serialized)
- No software pipelining (Tutorial 10)
- No TMA (Tutorial 08)
- Scattered C epilogue
- No predication for boundary tiles

On A100 with 512×512×256, expect 5–15 TFLOPS (vs. ~150 TFLOPS peak for cuBLAS with large matrices). The point is not speed — it's understanding the structure that every fast GEMM kernel shares.

> **Gotcha — dynamic vs. static shapes:** The global tensors use dynamic shapes (runtime M, N, K), but the smem layouts and tile shapes use static `Int<N>{}`. This is intentional — CuTe needs static shapes to generate efficient code (compile-time layout math, unrolled loops). Dynamic shapes in gmem are fine because the copy and MMA operate on the statically-shaped smem/register tiles.

> **Gotcha — smem capacity:** Two smem tiles of 128×32 halfs = 2 × 128 × 32 × 2 bytes = 16 KB. Well within SM80's 164 KB limit. But 128×128 tiles with BLK_K=64 would need 32 KB — still fine, but leaves less for double buffering. Always check: `smem_bytes = 2 × BLK_M × BLK_K × sizeof(half)`.

**What comes next:** This kernel serializes copy and compute — the Tensor Cores are idle while waiting for data loads. Tutorial 08 (The TMA Revolution) introduces hardware-accelerated copies that free the threads. Tutorial 10 (Pipeline Overlap) shows how to overlap copy and compute using ping-pong buffering, approaching peak throughput.
