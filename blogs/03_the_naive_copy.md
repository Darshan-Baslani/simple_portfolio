# The Naive Copy — Scalar vs. Vectorized Memory Movement

**Difficulty:** Beginner
**Prerequisites:** [Tutorial 01: Hello, Layout!](01_hello_layout.md), [Tutorial 02: The Art of Slicing](02_the_art_of_slicing.md), basic CUDA memory model

## 1. The Problem (The "Why")

You've partitioned your matrix beautifully. Every thread knows which elements it owns. Now it needs to actually *move* data — from global memory to shared memory, or from shared memory to registers. The simplest approach: a `for` loop that copies one `float` at a time.

```cpp
// The "obvious" way — one element per iteration
for (int i = 0; i < N; ++i) {
    dst[i] = src[i];
}
```

This works. It's also leaving **75% of your memory bandwidth on the table.**

Why? Because the GPU's memory bus is 128 bits wide. A single `float` is 32 bits. Every load instruction fetches 128 bits from DRAM anyway — but if you only asked for 32 bits, the other 96 bits are thrown away. You're paying for a 4-lane highway but driving in one lane.

The fix: **vectorized loads**. Instead of `LDG.32` (load 32 bits), the compiler emits `LDG.128` (load 128 bits = 4 floats at once). Same number of memory transactions, 4× the useful data per transaction.

CuTe handles this automatically — *if* your data meets two conditions. This tutorial shows you what those conditions are and how to make sure your copies always hit the fast path.

> **B200 Note:** On Hopper/Blackwell, vectorized copies are the minimum bar. The real prize is `cp.async` and TMA (Tutorials 04 and 08), which bypass registers entirely. But even those advanced paths require the same alignment and contiguity fundamentals covered here.

## 2. The Mental Model (The Visual)

### Scalar Copy: One Element at a Time

```text
Global Memory (128-bit bus):
┌─────────────────────────────────────┐
│  float₀  float₁  float₂  float₃   │  ← 128 bits fetched from DRAM
└─────────────────────────────────────┘
     ↓
   LDG.32 grabs only float₀ (32 bits)
   The other 96 bits? Wasted.

   Thread issues 4 separate loads:
   LDG.32 → float₀
   LDG.32 → float₁
   LDG.32 → float₂
   LDG.32 → float₃
   = 4 instructions, 4 transactions
```

### Vectorized Copy: Four Elements at Once

```text
Global Memory (128-bit bus):
┌─────────────────────────────────────┐
│  float₀  float₁  float₂  float₃   │  ← 128 bits fetched from DRAM
└─────────────────────────────────────┘
     ↓
   LDG.128 grabs ALL FOUR floats in one shot

   Thread issues 1 load:
   LDG.128 → float₀, float₁, float₂, float₃
   = 1 instruction, 1 transaction
```

### The Two Requirements for Vectorization

```text
Requirement 1: CONTIGUITY (stride-1)
   ✅ Contiguous:  [ f₀ | f₁ | f₂ | f₃ ]   stride = 1
                      +1   +1   +1
   ❌ Strided:     [ f₀ |    | f₁ |    ]   stride = 2
                      +2        +2           Can't bundle into one load

Requirement 2: ALIGNMENT (address divisible by vector width)
   ✅ Aligned:     addr 0x0000  →  [ f₀ f₁ f₂ f₃ ]   0 % 16 == 0
   ❌ Misaligned:  addr 0x0004  →  [ f₁ f₂ f₃ f₄ ]   4 % 16 ≠ 0
                   Can't do 128-bit load from a 32-bit boundary
```

> **The Dolly Rule:** Think of vectorization like moving boxes on a loading dock. `UniversalCopy` hand-carries one box at a time — always works, always slow. `AutoVectorizingCopy` is the smart worker who checks: "Are these boxes side-by-side (contiguous) and does the stack start at a dolly-sized slot (aligned)?" If yes, load them all onto the dolly in one trip. If not, fall back to one-at-a-time.

## 3. The Solution (The Code)

A CUDA kernel that copies a tile from global to shared memory using CuTe's `copy()`, then measures effective bandwidth for both scalar and vectorized paths.

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy.hpp>

#include <cstdio>
#include <chrono>

using namespace cute;

// ─── Kernel: copy a tile from global → shared memory ───
template <class CopyOp, class GmemLayout, class SmemLayout>
__global__ void copy_kernel(float const* __restrict__ g_ptr,
                            GmemLayout   gmem_layout,
                            SmemLayout   smem_layout)
{
    extern __shared__ float smem[];

    // Wrap raw pointers in CuTe Tensors
    auto g_tensor = make_tensor(make_gmem_ptr(g_ptr),  gmem_layout);  // (M, N)
    auto s_tensor = make_tensor(make_smem_ptr(smem),   smem_layout);  // (M, N)

    // Each thread gets its partition of the tile
    auto thr_g = local_partition(g_tensor, Layout<Shape<_8, _32>>{}, threadIdx.x);
    auto thr_s = local_partition(s_tensor, Layout<Shape<_8, _32>>{}, threadIdx.x);

    // ── The actual copy ──
    copy(CopyOp{}, thr_g, thr_s);
}

// ─── Benchmark harness ───
template <class CopyOp>
float benchmark_copy(const char* label, int M, int N, int iters)
{
    size_t bytes = M * N * sizeof(float);

    float *d_src;
    cudaMalloc(&d_src, bytes);
    cudaMemset(d_src, 1, bytes);  // fill with something

    // Layouts: column-major for both gmem and smem
    auto gmem_layout = make_layout(make_shape(M, N), LayoutLeft{});
    auto smem_layout = make_layout(make_shape(M, N), LayoutLeft{});

    int smem_bytes = M * N * sizeof(float);
    dim3 block(256);
    dim3 grid(1);

    // Warmup
    copy_kernel<CopyOp><<<grid, block, smem_bytes>>>(d_src, gmem_layout, smem_layout);
    cudaDeviceSynchronize();

    // Timed loop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        copy_kernel<CopyOp><<<grid, block, smem_bytes>>>(d_src, gmem_layout, smem_layout);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float gb_per_sec = (float(bytes) * iters / 1e9) / (ms / 1e3);
    printf("%-40s  %8.2f GB/s   (%6.3f ms for %d iters)\n", label, gb_per_sec, ms, iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    return gb_per_sec;
}

int main()
{
    int M = 128, N = 256, iters = 1000;

    printf("Copying %dx%d tile (%zu bytes) from Global → Shared Memory\n\n", M, N, M*N*sizeof(float));

    // 1. Scalar: UniversalCopy copies one float (32 bits) per call
    benchmark_copy<UniversalCopy<float>>(
        "UniversalCopy<float>  (scalar, 32-bit)", M, N, iters);

    // 2. Vectorized: AutoVectorizingCopy bundles up to 128 bits
    benchmark_copy<AutoVectorizingCopyWithAssumedAlignment<128>>(
        "AutoVectorizingCopy   (vector, 128-bit)", M, N, iters);

    // 3. For reference: explicit 64-bit vectorization
    benchmark_copy<AutoVectorizingCopyWithAssumedAlignment<64>>(
        "AutoVectorizingCopy   (vector,  64-bit)", M, N, iters);

    return 0;
}
```

**Expected Output (approximate — numbers vary by GPU):**

```text
Copying 128x256 tile (131072 bytes) from Global → Shared Memory

UniversalCopy<float>  (scalar, 32-bit)      85.20 GB/s   ( 1.539 ms for 1000 iters)
AutoVectorizingCopy   (vector, 128-bit)     310.45 GB/s   ( 0.422 ms for 1000 iters)
AutoVectorizingCopy   (vector,  64-bit)     205.30 GB/s   ( 0.638 ms for 1000 iters)
```

The vectorized 128-bit copy is roughly **3–4× faster** than scalar. The 64-bit variant lands in between.

## 4. Step-by-Step Explanation

**Line: `auto g_tensor = make_tensor(make_gmem_ptr(g_ptr), gmem_layout);`**

- `make_gmem_ptr(g_ptr)` wraps the raw pointer and tags it as global memory. CuTe uses this tag to select the right load instruction (`LDG` vs. `LDS` vs. register move). Without it, CuTe can't distinguish memory spaces.

**Line: `auto thr_g = local_partition(g_tensor, Layout<Shape<_8, _32>>{}, threadIdx.x);`**

- The thread layout `(8, 32)` means 8 rows × 32 columns = 256 threads. This is the `thr_layout` from Tutorial 02. Each thread gets a `(M/8, N/32)` = `(16, 8)` subtensor — 128 floats to copy.

**Line: `copy(CopyOp{}, thr_g, thr_s);`**

- This is the core call. CuTe dispatches based on `CopyOp`:
  - **`UniversalCopy<float>`**: Copies one `float` at a time. The inner loop is `dst(i) = src(i)` — a simple scalar assignment. The compiler emits `LDG.32` + `STS.32` per element.
  - **`AutoVectorizingCopyWithAssumedAlignment<128>`**: CuTe inspects the source and destination layouts at compile time and asks three questions:

```text
   1. max_common_vector(src, dst):  How many elements are contiguous
                                    (stride-1) in BOTH tensors?
                                    For column-major (16,8):(1,16) → 16 elements.

   2. max_alignment(src, dst):      What's the natural alignment?
                                    GCD of shape and stride components.

   3. gcd(contiguous_bits, alignment_bits, MaxVecBits):
                                    The actual vector width to use.
                                    min(16 * 32, alignment, 128) = 128 bits.
```

  When the result is 128 bits and both tensors qualify, CuTe calls `recast<uint128_t>(tensor)` to reinterpret 4 floats as a single 128-bit integer, then copies those. The compiler emits `LDG.128` + `STS.128` — one instruction moves 4 floats.

**What happens when vectorization fails:**

If you use a strided layout (e.g., stride-2 between elements), `max_common_vector` returns 1. CuTe falls back to element-by-element copy — same speed as `UniversalCopy`. Your code doesn't crash, it just silently takes the slow path. This is why understanding contiguity matters.

**Line: `copy(src, dst)` (no CopyOp argument)**

- When you call `copy()` *without* specifying a copy operation, CuTe picks one for you:
  - Static layout? → `AutoVectorizingCopyWithAssumedAlignment<128>` (assumes 128-bit aligned pointers)
  - Dynamic layout? → `AutoVectorizingCopyWithAssumedAlignment<8>` (conservative — no alignment assumed)
- **Takeaway:** The default `copy(src, dst)` already tries to vectorize. You don't need to pass `AutoVectorizingCopy` explicitly unless you want to control `MaxVecBits`.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** Copying data is like **moving boxes on a loading dock**. `UniversalCopy` hand-carries one box at a time — it always works, but it's slow. `AutoVectorizingCopy` is the smart worker who checks: "Are these boxes side-by-side on the shelf (contiguous, stride-1)? Does the stack start at a slot that fits my dolly (aligned to 128-bit boundary)?" If yes, load them onto the dolly and move 4 boxes in one trip. If not, fall back to hand-carrying.

**The Three Laws of Auto-Vectorization:**

1. **Contiguity** — Both source and destination must have stride-1 elements in memory. Column-major `(M,N):(1,M)` is contiguous along mode-0. Row-major `(M,N):(N,1)` is contiguous along mode-1. Irregular strides (e.g., stride-3) → scalar fallback.
2. **Alignment** — The starting address must be divisible by the vector width. `cudaMalloc` returns 256-byte-aligned pointers, so global memory is fine. Shared memory offsets can break alignment if your tile size isn't a multiple of 4 floats (16 bytes).
3. **MaxVecBits** — CuTe caps vectorization at this value (default 128 bits). You can lower it for testing, but 128 is the sweet spot for most GPU architectures.

**The Copy Atom Hierarchy:**

| Copy Operation | Vector Width | When to Use |
| :--- | :--- | :--- |
| `UniversalCopy<T>` | `sizeof(T)` bits | Fallback. Always works, never fast. |
| `AutoVectorizingCopyWithAssumedAlignment<128>` | Up to 128 bits | Default for static layouts. The "just works" choice. |
| `AutoVectorizingCopyWithAssumedAlignment<64>` | Up to 64 bits | When you know alignment is only 8-byte. |
| `AutoVectorizingCopyWithAssumedAlignment<8>` | Up to 8 bits | Conservative — CuTe uses this for dynamic layouts. Essentially scalar. |

**Hardware Note:** The GPU memory bus fetches 128 bits (or 32 bytes on some architectures) per transaction regardless of how much you asked for. Scalar loads waste bus bandwidth. Vectorized loads use the full transaction. This is why `LDG.128` doesn't cost more latency than `LDG.32` — the memory controller does the same work either way. You're just *using* more of what it already fetched.

> **Gotcha:** CuTe's auto-vectorization is a **compile-time** decision. If your layouts are dynamic (runtime shapes/strides), CuTe can't prove contiguity at compile time and falls back to scalar. Use `Int<N>{}` (static integers) for shapes and strides whenever possible — it's not just about performance, it's about enabling vectorization.

**What comes next:** This tutorial copies with *one thread per partition*. But moving a 128×128 tile with 256 threads — where all threads cooperate on the same tile — requires coordinating who copies what. That's `TiledCopy` (Tutorial 04), which combines a `Copy_Atom` (the vectorization from this tutorial) with a thread layout and a value layout to orchestrate the full tile copy.
