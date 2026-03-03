# Hello, MMA — Your First Tensor Core Instruction

**Difficulty:** Intermediate
**Prerequisites:** [Tutorial 01: Hello, Layout!](01_hello_layout.md), [Tutorial 04: The Parallel Copy](04_the_parallel_copy.md) (same partition API pattern)

## 1. The Problem (The "Why")

Tutorials 03–05 were about *moving data* — getting floats from global memory into shared memory, fast and conflict-free. But at some point, you need to actually *compute*. On modern NVIDIA GPUs, the fastest math unit isn't the CUDA core — it's the **Tensor Core**.

A Tensor Core executes a matrix multiply-accumulate (MMA) as a single hardware instruction. The SM80 `mma.sync.aligned.m16n8k16` instruction multiplies a 16×16 half-precision matrix by an 8×16 half-precision matrix and accumulates into a 16×8 float matrix — **2048 multiply-adds in one clock cycle**, across all 32 threads of a warp.

The catch: you can't just pass pointers. The hardware expects A, B, and C fragments to be distributed across 32 threads in a **very specific register layout**. Thread 0 holds certain rows and columns of A, thread 7 holds others, and the MMA instruction reaches into all 32 threads' registers simultaneously. Get the distribution wrong and you get garbage — or a hardware trap.

CuTe's `TiledMMA` handles this distribution for you, using the **exact same partition pattern** you learned with `TiledCopy`:

| TiledCopy (Tutorial 04) | TiledMMA (This Tutorial) |
| :--- | :--- |
| `make_tiled_copy(Copy_Atom, thr, val)` | `make_tiled_mma(MMA_Atom<Op>{})` |
| `get_thread_slice(tid)` | `get_thread_slice(tid)` |
| `partition_S` / `partition_D` | `partition_A` / `partition_B` / `partition_C` |
| `copy(tiled_copy, src, dst)` | `gemm(tiled_mma, fragA, fragB, fragC)` |

Same API shape, different hardware backend. If you understood Tutorial 04's ownership maps and partitions, you already know 80% of this tutorial.

> **B200 Note:** On Hopper/Blackwell, the same `TiledMMA` pattern drives WGMMA (Warpgroup MMA) — 128 threads feeding a larger Tensor Core instruction. Tutorial 09 covers that. The API is identical; only the atom changes.

## 2. The Mental Model (The Visual)

### The Stamping Press

Think of a Tensor Core as a **stamping press** in a factory. You load raw-material trays (A and B fragments) into specific slots, press the button, and out comes the product (C accumulator). The press is fixed-size — it always stamps a 16×8 tile of results from 16×16 and 8×16 inputs.

```text
                        ┌─────────────────────────────┐
                        │      STAMPING PRESS          │
                        │   (One MMA Instruction)      │
                        │                              │
   ┌──────────────┐     │                              │     ┌──────────┐
   │  A (16×16)   │────▶│   32 threads cooperate:      │────▶│ C (16×8) │
   │   half       │     │   each loads its tray slot,  │     │  float   │
   │  256 values  │     │   hardware does the rest     │     │ 128 vals │
   └──────────────┘     │                              │     └──────────┘
   ┌──────────────┐     │   2048 multiply-adds         │
   │  B (8×16)    │────▶│   in ONE clock cycle         │
   │   half       │     │                              │
   │  128 values  │     └─────────────────────────────-┘
   └──────────────┘
```

### Per-Thread Fragment Sizes

The 256 values of A, 128 values of B, and 128 values of C are split evenly across 32 threads:

```text
Thread #7 (one of 32):
┌────────────────────────────────────────────────────┐
│  Registers:                                        │
│    A fragment: 8 half values   (256 total / 32)    │
│    B fragment: 4 half values   (128 total / 32)    │
│    C fragment: 4 float values  (128 total / 32)    │
│                                                    │
│  The hardware knows exactly which matrix cells     │
│  these 8+4+4 values correspond to.                 │
│  CuTe knows too — that's what partition_A/B/C do.  │
└────────────────────────────────────────────────────┘
```

You don't need to know the exact register-to-coordinate mapping (it's in the PTX docs if you're curious). CuTe's partition functions handle it transparently — just like `TiledCopy` handles thread-to-element mapping without you memorizing a table.

### The API Flow

```text
make_tiled_mma(MMA_Atom<Op>{})
         │
    get_thread_slice(tid)
         │
    ┌────┴──────────────────┬──────────────────┐
    │                       │                  │
partition_A(sA)       partition_B(sB)     partition_C(gC)
    │                       │                  │
make_fragment_like    make_fragment_like   make_fragment_like
    │                       │                  │
copy(smem → reg)      copy(smem → reg)     clear(accum)
    │                       │                  │
    └───────────┬───────────┘                  │
                │                              │
        gemm(tiled_mma, fragA, fragB, accum)───┘
                              │
                     copy(accum → gC)
```

## 3. The Solution (The Code)

Two kernels: first, an ownership map that shows which thread owns which C element (same technique as Tutorial 04). Second, the actual micro-GEMM: load A and B into shared memory, partition them for the MMA, execute one Tensor Core instruction, and write C back to global memory.

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cuda_fp16.h>
#include <cstdio>

using namespace cute;

// ─── Kernel 1: MMA Ownership Map ───
// Which thread owns which elements of the C matrix?
__global__ void mma_ownership_map()
{
    using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
    auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Op>{});

    // Shared memory to stamp thread IDs into C positions
    __shared__ int smem_C[16 * 8];
    auto sC_layout = make_layout(make_shape(Int<16>{}, Int<8>{}));
    auto sC = make_tensor(make_smem_ptr(smem_C), sC_layout);

    // Clear
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        smem_C[i] = -1;
    __syncthreads();

    // Each thread stamps its ID into its owned C cells
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsC = thr_mma.partition_C(sC);

    for (int i = 0; i < size(tCsC); ++i)
        tCsC(i) = threadIdx.x;
    __syncthreads();

    // Thread 0 prints the map
    if (threadIdx.x == 0) {
        printf("MMA C Ownership Map (16x8) — SM80_16x8x16_F32F16F16F32_TN\n");
        printf("Each thread holds 4 float accumulators.\n\n");

        printf("       ");
        for (int n = 0; n < 8; ++n) printf("n=%-4d", n);
        printf("\n");

        for (int m = 0; m < 16; ++m) {
            printf("m=%-4d ", m);
            for (int n = 0; n < 8; ++n)
                printf("T%02d   ", sC(m, n));
            printf("\n");
        }
    }
}

// ─── Kernel 2: Micro-GEMM ───
// One warp computes C(16×8) += A(16×16) × B(8×16)^T using a single MMA instruction.
__global__ void micro_gemm(half const* A_ptr, half const* B_ptr, float* C_ptr)
{
    using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;
    auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Op>{});

    // ── Shared memory ──
    __shared__ half smem_A[16 * 16];   // M × K
    __shared__ half smem_B[8 * 16];    // N × K

    // A layout: (M,K) = (16,16), K stride-1 (required by TN atom)
    auto sA_layout = make_layout(make_shape(Int<16>{}, Int<16>{}),
                                 make_stride(Int<16>{}, Int<1>{}));
    // B layout: (N,K) = (8,16), K stride-1
    auto sB_layout = make_layout(make_shape(Int<8>{}, Int<16>{}),
                                 make_stride(Int<16>{}, Int<1>{}));
    // C layout: (M,N) = (16,8), column-major
    auto gC_layout = make_layout(make_shape(Int<16>{}, Int<8>{}));

    auto sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem_B), sB_layout);
    auto gC = make_tensor(make_gmem_ptr(C_ptr),  gC_layout);

    // ── 1. Load A, B from global to shared (simple loop — not TiledCopy) ──
    auto gA = make_tensor(make_gmem_ptr(A_ptr), sA_layout);
    auto gB = make_tensor(make_gmem_ptr(B_ptr), sB_layout);

    for (int i = threadIdx.x; i < size(sA); i += blockDim.x)
        sA(i) = gA(i);
    for (int i = threadIdx.x; i < size(sB); i += blockDim.x)
        sB(i) = gB(i);
    __syncthreads();

    // ── 2. Partition A, B, C for this thread ──
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tCsA = thr_mma.partition_A(sA);    // (MMA, MMA_M, MMA_K) = (8, 1, 1)
    auto tCsB = thr_mma.partition_B(sB);    // (MMA, MMA_N, MMA_K) = (4, 1, 1)
    auto tCgC = thr_mma.partition_C(gC);    // (MMA, MMA_M, MMA_N) = (4, 1, 1)

    // ── 3. Create register fragments ──
    auto tCrA = make_fragment_like(tCsA);   // 8 half regs for A
    auto tCrB = make_fragment_like(tCsB);   // 4 half regs for B
    auto accum = make_fragment_like(tCgC);  // 4 float regs for C
    clear(accum);

    // ── 4. Copy A, B from shared memory to registers ──
    copy(tCsA, tCrA);
    copy(tCsB, tCrB);

    // ── 5. Execute the MMA! ──
    gemm(tiled_mma, tCrA, tCrB, accum);

    // ── 6. Write result back to global memory ──
    copy(accum, tCgC);
}

int main()
{
    constexpr int M = 16, N = 8, K = 16;

    // ─── 1. Ownership Map ───
    printf("=== MMA Ownership Map ===\n\n");
    mma_ownership_map<<<1, 32>>>();
    cudaDeviceSynchronize();

    // ─── 2. Micro-GEMM ───
    printf("\n=== Micro-GEMM: C(16x8) = A(16x16) * B(8x16)^T ===\n\n");

    // Test data: A = all ones, B = all ones → C[m][n] = K = 16.0
    half h_A[M * K], h_B[N * K];
    float h_C[M * N] = {};

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < N * K; ++i) h_B[i] = __float2half(1.0f);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice);

    micro_gemm<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // C is column-major: element (m,n) is at offset m + M*n
    printf("C (16x8) — each entry = K * 1.0 * 1.0 = 16.0:\n\n");
    printf("       ");
    for (int n = 0; n < N; ++n) printf("n=%-6d", n);
    printf("\n");
    for (int m = 0; m < M; ++m) {
        printf("m=%-4d ", m);
        for (int n = 0; n < N; ++n)
            printf("%-8.1f", h_C[m + M * n]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
```

**Expected Output:**

```text
=== MMA Ownership Map ===

MMA C Ownership Map (16x8) — SM80_16x8x16_F32F16F16F32_TN
Each thread holds 4 float accumulators.

       n=0   n=1   n=2   n=3   n=4   n=5   n=6   n=7
m=0    T00   T00   T04   T04   T08   T08   T12   T12
m=1    T01   T01   T05   T05   T09   T09   T13   T13
m=2    T02   T02   T06   T06   T10   T10   T14   T14
m=3    T03   T03   T07   T07   T11   T11   T15   T15
m=4    T16   T16   T20   T20   T24   T24   T28   T28
m=5    T17   T17   T21   T21   T25   T25   T29   T29
m=6    T18   T18   T22   T22   T26   T26   T30   T30
m=7    T19   T19   T23   T23   T27   T27   T31   T31
m=8    T00   T00   T04   T04   T08   T08   T12   T12
m=9    T01   T01   T05   T05   T09   T09   T13   T13
m=10   T02   T02   T06   T06   T10   T10   T14   T14
m=11   T03   T03   T07   T07   T11   T11   T15   T15
m=12   T16   T16   T20   T20   T24   T24   T28   T28
m=13   T17   T17   T21   T21   T25   T25   T29   T29
m=14   T18   T18   T22   T22   T26   T26   T30   T30
m=15   T19   T19   T23   T23   T27   T27   T31   T31

=== Micro-GEMM: C(16x8) = A(16x16) * B(8x16)^T ===

C (16x8) — each entry = K * 1.0 * 1.0 = 16.0:

       n=0     n=1     n=2     n=3     n=4     n=5     n=6     n=7
m=0    16.0    16.0    16.0    16.0    16.0    16.0    16.0    16.0
m=1    16.0    16.0    16.0    16.0    16.0    16.0    16.0    16.0
...
m=15   16.0    16.0    16.0    16.0    16.0    16.0    16.0    16.0
```

The ownership map reveals the hardware's register distribution: threads 0–3 and 16–19 alternate in 4-row blocks along M, and each thread owns pairs of adjacent columns. Thread T00 holds C[0][0], C[0][1], C[8][0], C[8][1] — four floats scattered across two 4-row blocks. You don't need to memorize this pattern; `partition_C` handles it automatically.

The GEMM result is 16.0 everywhere: each entry is the dot product of a length-16 all-ones row from A with a length-16 all-ones row from B. One instruction, 2048 multiply-adds, 32 threads cooperating through their registers.

## 4. Step-by-Step Explanation

**Line: `using MMA_Op = SM80_16x8x16_F32F16F16F32_TN;`**

This names the PTX instruction we want: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. The naming convention:

| Part | Meaning |
| :--- | :--- |
| `SM80` | Ampere architecture (compute capability 8.0+) |
| `16x8x16` | M × N × K — the tile dimensions of one MMA instruction |
| `F32F16F16F32` | Types: D=float, A=half, B=half, C=float (D, A, B, C order) |
| `TN` | A is K-major (Transposed), B is K-major (Normal) |

The "TN" layout is the most common: both A and B have their K dimension as stride-1. This means your shared memory layouts for A(M,K) and B(N,K) should both have K as the fast-moving axis.

**Line: `auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Op>{});`**

Wraps the raw hardware instruction in CuTe's `MMA_Atom`, then promotes it to a `TiledMMA`. With no extra tiling arguments, this creates the simplest possible MMA — one atom, one warp, no repetition. The tile size is exactly 16×8×16.

Compare with Tutorial 04's `make_tiled_copy(Copy_Atom, thr_layout, val_layout)`. For MMA, the thread layout and value layout are determined by the hardware instruction itself — there's nothing to choose. The atom *is* the complete plan.

**Lines: `sA_layout` with `make_stride(Int<16>{}, Int<1>{})`**

A is (M, K) = (16, 16) with K stride-1. This means `sA(m, k) = m * 16 + k * 1` — moving along K is contiguous in memory. The TN atom requires this: the PTX instruction's "row" descriptor for A means K is the fast index.

If you accidentally used column-major (M stride-1, K stride-16), the MMA would read the wrong values — you'd be feeding columns of A where it expects rows.

**Lines: `sB_layout` with `make_stride(Int<16>{}, Int<1>{})`**

B is (N, K) = (8, 16), also K stride-1. Same reason: the TN atom's "col" descriptor for B also means K is the fast index. Both A and B want K contiguous.

> This is why "TN" is popular: when A and B are both K-major, your global memory loads can be coalesced along the K dimension. You'll see this pattern again in Tutorial 07's full GEMM.

**Line: `auto tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K) = (8, 1, 1)`**

`partition_A` asks: "Given the full shared memory tensor `sA` and the MMA's register layout, which elements belong to this thread?" It returns a 3-mode tensor:

- **Mode 0 (MMA):** The 8 half values this thread feeds into one atom invocation.
- **Mode 1 (MMA_M):** How many times the atom repeats along M. Our A has M=16 and the atom handles M=16, so 1 repetition.
- **Mode 2 (MMA_K):** How many times the atom repeats along K. Our K=16 matches the atom's K=16, so 1 repetition.

Compare with Tutorial 04: `partition_S` returned `(ValuesPerAtom, Reps...)`. Same idea, different dimension names.

**Line: `auto tCrA = make_fragment_like(tCsA);`**

Creates a register tensor with the same shape as the smem partition: `(8, 1, 1)` of half-precision values. These 8 registers are this thread's personal tray — the slot in the stamping press that it's responsible for loading.

**Line: `copy(tCsA, tCrA);`**

Copies this thread's 8 half values from shared memory to registers. Each thread reads from different smem addresses (determined by `partition_A`), so there's no conflict. After this line, all 32 threads have their A fragments loaded and ready.

**Line: `gemm(tiled_mma, tCrA, tCrB, accum);`**

This is the magic line. `gemm()` dispatches the PTX `mma.sync` instruction, which:

1. Reads A fragments from all 32 threads' registers (8 halfs each = 256 total = 16×16 matrix).
2. Reads B fragments from all 32 threads' registers (4 halfs each = 128 total = 8×16 matrix).
3. Multiplies the 16×16 matrix by the 8×16 matrix (as C += A × B^T).
4. Accumulates the 16×8 result into all 32 threads' C registers (4 floats each = 128 total).

All in one clock cycle. The `.sync` in the PTX name means all threads in the warp participate simultaneously — this is a collective operation, not a per-thread one.

**Line: `copy(accum, tCgC);`**

Copies the 4 float accumulator values from registers to their corresponding positions in global memory C. Each thread writes to different addresses (determined by `partition_C`), so no conflicts. The host then reads back the complete 16×8 result.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** `MMA_Atom` is a **stamping press** — a fixed-size hardware machine that takes raw-material trays (A and B register fragments), processes them all at once, and stamps out a product (C accumulator). Each worker (thread) loads their assigned tray slot. `TiledMMA` is the factory floor plan that coordinates the workers. `gemm()` is pressing the button.

This analogy extends the Tutorial 03–04 warehouse metaphor: data arrives from the truck (global memory) via the loading dock (TiledCopy + smem), and now the stamping press (TiledMMA) processes it in the factory. The same workers (threads) do both jobs — they just switch from mover role to machine-operator role.

**Fragment size formula:**

For any MMA atom with shape M×N×K using W threads:

| Fragment | Per-thread values | Total |
| :--- | :--- | :--- |
| A | M × K / W | M × K |
| B | N × K / W | N × K |
| C | M × N / W | M × N |

For SM80_16x8x16 with W=32: A = 8 halfs, B = 4 halfs, C = 4 floats.

**What "TN" means for your layouts:**

| Layout | A (M, K) | B (N, K) |
| :--- | :--- | :--- |
| **TN** | K stride-1 `(K, 1)` | K stride-1 `(K, 1)` |
| TT | K stride-1 `(K, 1)` | N stride-1 `(1, N)` |
| NN | M stride-1 `(1, M)` | N stride-1 `(1, N)` |
| NT | M stride-1 `(1, M)` | K stride-1 `(K, 1)` |

TN is the most common because both A and B have K contiguous — global memory loads along the K-reduction dimension are coalesced.

**The ownership map tells you something important:**

Thread T00 owns C[0][0], C[0][1], C[8][0], C[8][1]. These four values are *not* contiguous in column-major memory (offsets 0, 16, 8, 24). This means writing the C accumulator to global memory via `copy(accum, tCgC)` produces scattered stores — each thread writes 4 separate floats to non-contiguous addresses. For a real GEMM kernel (Tutorial 07), the epilogue stage would use a TiledCopy to reorganize the writes through shared memory for coalesced global stores.

**Hardware note — checking your GPU:**

The SM80 MMA atom requires compute capability 8.0 or higher (A100, RTX 3090, etc.). To check:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Should print 8.0 or higher
```

**Hardware note — throughput:**

On A100, one SM can issue one `mma.sync.m16n8k16` per cycle per warp scheduler. With 4 warp schedulers, that's 4 × 2048 = 8192 FMAs/cycle per SM. Multiply by 108 SMs and a 1.41 GHz clock: ~2.5 TFLOPS of Tensor Core throughput *per SM schedule*. But you'll only reach this if the data pipeline (Tutorials 03–05) keeps up.

> **Gotcha — layout mismatch:** If your smem layout doesn't match the TN convention (K stride-1), the MMA will read the wrong elements. The symptom is silently wrong results, not a crash. Always double-check that your A and B shared memory strides put K as stride-1 for TN atoms.

> **Gotcha — half precision:** The input matrices *must* be `half` (F16). If you accidentally store `float` values in smem and cast the pointer to `half*`, you'll get garbage. Use `__float2half()` to convert before storing.

**What comes next:** This tutorial computed a single 16×8 tile using one MMA instruction. But real matrices are much larger than 16×8. Tutorial 07 (The Global GEMM) shows how to tile over M, N, and K — looping the MMA across tiles, piping data from global memory through shared memory into the Tensor Core in a continuous stream.
