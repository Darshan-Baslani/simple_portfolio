# Swizzling — Avoiding Shared Memory Bank Conflicts

**Difficulty:** Intermediate
**Prerequisites:** [Tutorial 01: Hello, Layout!](01_hello_layout.md), [Tutorial 04: The Parallel Copy](04_the_parallel_copy.md), basic shared memory understanding

## 1. The Problem (The "Why")

Tutorial 04's `TiledCopy` moves data from global to shared memory — 32 threads, 128-bit vectorized stores, zero coordination issues. Beautiful.

But there's a hidden trap. Shared memory isn't one big flat buffer. It's split into **32 banks**, each serving one 4-byte access per cycle. When two threads in the same warp access the *same bank* at the *same time* (different addresses within that bank), the hardware serializes their accesses. This is a **bank conflict**. An N-way conflict means N sequential accesses instead of 1 — your shared memory bandwidth drops to 1/N.

Here's the catch: bank conflicts depend on the **access pattern**, not just the data layout. Your `TiledCopy` might *write* to smem conflict-free (column-first, nicely spread across banks). But the next stage — an MMA reading that data row-first — might collide on every access.

The culprit is regularity. Column-major strides like 8, 16, 32, or 128 divide evenly into 32 banks, so different columns in the same row keep landing on the same bank. The fix: **swizzle** the shared memory layout. CuTe's `Swizzle<B, M, S>` XORs parts of the address to break this regularity — one line of code, zero bank conflicts.

> **B200 Note:** On Blackwell (SM100), all supported MMA swizzle modes — including no-swizzle (8×16B interleaved) — are **bank-conflict-free on both the MMA read side and the TMA write side**. Swizzling still matters, though: using no-swizzle or smaller swizzle modes can reduce **TMA achievable throughput** when populating shared memory. So on Hopper/Blackwell the swizzle is primarily about maximizing TMA write bandwidth, not avoiding MMA read conflicts.

## 2. The Mental Model (The Visual)

### How Shared Memory Banks Work

Shared memory is divided into 32 banks, each 4 bytes wide. Bank assignment is cyclic:

```text
Float index:    0    1    2    3    4   ...   31   32   33  ...
Bank:          B00  B01  B02  B03  B04 ...  B31  B00  B01  ...

                    bank = float_index % 32
```

Within a single warp (32 threads executing simultaneously):
- Each thread accesses a *different* bank → **1 cycle** (full parallelism)
- K threads access the *same* bank → **K cycles** (serialized)

### The Problem: Column-Major + Row Access = Conflict

Consider an 8×8 tile stored column-major in shared memory (stride = `(1, 8)`):

```text
WITHOUT Swizzle — Bank map for 8×8 column-major (1,8):

       c0    c1    c2    c3    c4    c5    c6    c7
r0     B00   B08   B16   B24   B00   B08   B16   B24    ← 2-way conflict!
r1     B01   B09   B17   B25   B01   B09   B17   B25      c0 & c4 share B00
r2     B02   B10   B18   B26   B02   B10   B18   B26      c1 & c5 share B08
r3     B03   B11   B19   B27   B03   B11   B19   B27      c2 & c6 share B16
r4     B04   B12   B20   B28   B04   B12   B20   B28      c3 & c7 share B24
r5     B05   B13   B21   B29   B05   B13   B21   B29
r6     B06   B14   B22   B30   B06   B14   B22   B30
r7     B07   B15   B23   B31   B07   B15   B23   B31
```

Reading **down a column** (the TiledCopy write path): banks 0,1,2,3,4,5,6,7 — all different. No conflict.

Reading **across a row** (e.g., MMA consuming data): row 0 hits banks 0,8,16,24,**0**,8,16,24 — **2-way conflict** on every bank! Columns 0 and 4 always collide, 1 and 5 collide, and so on.

Why? Because the column stride (8) divides evenly into 32: `(col × 8) % 32` cycles with period 4, so columns 0 and 4 produce the same bank. The regularity of the stride creates a repeating pattern that piles threads onto the same banks.

> With larger strides the problem gets worse. A stride of 32 maps *every column in the same row* to the same bank — an 8-way conflict for 8 columns. A stride of 128? Still hits the same bank pattern, just at a bigger scale.

### The Fix: Swizzle Breaks the Pattern

`Swizzle<3, 2, 3>` XORs high address bits into low address bits, scrambling the bank assignment so no two columns in the same row share a bank:

```text
WITH Swizzle<3,2,3> — Bank map for the same 8×8 tile:

       c0    c1    c2    c3    c4    c5    c6    c7
r0     B00   B08   B16   B24   B04   B12   B20   B28    ← all 8 unique!
r1     B01   B09   B17   B25   B05   B13   B21   B29
r2     B02   B10   B18   B26   B06   B14   B22   B30
r3     B03   B11   B19   B27   B07   B15   B23   B31
r4     B04   B12   B20   B28   B00   B08   B16   B24    ← all 8 unique!
r5     B05   B13   B21   B29   B01   B09   B17   B25
r6     B06   B14   B22   B30   B02   B10   B18   B26
r7     B07   B15   B23   B31   B03   B11   B19   B27
```

Read across row 0: banks 0,8,16,24,**4**,12,20,28 — **all 8 unique!**
Read across row 4: banks 4,12,20,28,**0**,8,16,24 — **all 8 unique!**
Read down column 4: banks 4,5,6,7,0,1,2,3 — **all 8 unique!**

Zero bank conflicts in any direction. The XOR shifted columns 4–7 by 4 banks relative to columns 0–3, breaking the collision pattern.

### The Brick Wall Analogy

```text
Without swizzle (joints aligned):        With swizzle (staggered):

┌────┬────┬────┬────┐                    ┌────┬────┬────┬────┐
│ B0 │ B8 │ B0 │ B8 │                    │ B0 │ B8 │ B4 │B12 │
├────┼────┼────┼────┤                    ├────┼────┼────┼────┤
│ B0 │ B8 │ B0 │ B8 │                    │ B4 │B12 │ B0 │ B8 │
├────┼────┼────┼────┤                    ├────┼────┼────┼────┤
│ B0 │ B8 │ B0 │ B8 │                    │ B0 │ B8 │ B4 │B12 │
└────┴────┴────┴────┘                    └────┴────┴────┴────┘
Same banks every row                     Banks shift per row
→ conflicts stack up                     → conflicts eliminated
```

Swizzle is the GPU equivalent of **staggered brick-laying**. In a brick wall, each row is offset by half a brick so the joints don't line up — this prevents cracks from running straight through. In shared memory, each row's addresses are XOR-shifted so the bank assignments don't repeat — this prevents bank conflicts from stacking up.

### How `Swizzle<B, M, S>` Works

The swizzle modifies a flat address by XOR-ing two groups of bits:

```text
Address bit layout for Swizzle<3, 2, 3>:

  bit:   7  6  5 │ 4  3  2 │ 1  0
         ────────│─────────│──────
         source  │ target  │ free
         (B=3)   │ (B=3)   │(M=2)
                 │    ↑    │
                 └────XOR──┘
                   (shift S=3)

  swizzled = addr ^ ((addr >> S) & mask)

  where mask covers B bits at the target position.
```

The three parameters:

| Parameter | Meaning | Effect |
| :--- | :--- | :--- |
| **M** (free bits) | Bottom M bits are untouched | 2^M elements stay contiguous. **M=2 → 4 floats (128 bits) stay together → vectorized loads still work!** |
| **B** (XOR width) | Number of bits to XOR | Scrambles across 2^B banks at a time. B=3 → 8-bank groups. |
| **S** (shift) | Distance between source and target bit fields | Target bits = `[M : M+B)`. Source bits = `[M+S : M+S+B)`. |

The critical parameter is **M**: it controls the granularity of the swizzle. Because bits `[0:M)` are untouched, blocks of 2^M consecutive elements remain contiguous after swizzling. With M=2, blocks of 4 floats (= 128 bits) are preserved — exactly what `LDG.128` / `STS.128` needs.

## 3. The Solution (The Code)

A bank conflict visualizer that prints the bank assignment map with and without swizzle, followed by a `TiledCopy` demonstrating that the swizzle is transparent to the copy.

```cpp
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

#include <cstdio>

using namespace cute;

// ─── Kernel: Print bank assignment for every tile cell ───
template <class Layout>
__global__ void bank_conflict_visualizer(Layout smem_layout, const char* label)
{
    if (threadIdx.x != 0) return;

    int M = size<0>(smem_layout);
    int N = size<1>(smem_layout);

    // ── Print bank map ──
    printf("%s (%d x %d):\n\n", label, M, N);

    printf("       ");
    for (int n = 0; n < N; ++n) printf("c%-4d ", n);
    printf("\n");

    for (int m = 0; m < M; ++m) {
        printf("r%-4d  ", m);
        for (int n = 0; n < N; ++n) {
            int addr = smem_layout(m, n);   // flat offset (in floats)
            int bank = addr % 32;
            printf("B%02d   ", bank);
        }
        printf("\n");
    }

    // ── Count row-wise conflicts ──
    int total_conflicts = 0;
    for (int m = 0; m < M; ++m) {
        int bank_hits[32] = {};
        for (int n = 0; n < N; ++n) {
            bank_hits[smem_layout(m, n) % 32]++;
        }
        for (int b = 0; b < 32; ++b) {
            if (bank_hits[b] > 1)
                total_conflicts += bank_hits[b] - 1;
        }
    }
    printf("\nRow-wise bank conflicts: %d  (%s)\n\n",
           total_conflicts, total_conflicts == 0 ? "CLEAN" : "CONFLICTS!");
}

// ─── Kernel: TiledCopy into swizzled smem ───
template <class TiledCopy, class GmemLayout, class SmemLayout>
__global__ void copy_with_swizzle(float const* __restrict__ g_ptr,
                                  GmemLayout   gmem_layout,
                                  SmemLayout   smem_layout,
                                  TiledCopy    tiled_copy)
{
    extern __shared__ float smem[];

    auto g_tensor = make_tensor(make_gmem_ptr(g_ptr), gmem_layout);
    auto s_tensor = make_tensor(make_smem_ptr(smem),  smem_layout);

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_g    = thr_copy.partition_S(g_tensor);
    auto thr_s    = thr_copy.partition_D(s_tensor);

    // ── Copy — swizzle is completely transparent ──
    copy(tiled_copy, thr_g, thr_s);

    __syncthreads();

    // Thread 0 verifies the data (reading through the swizzled layout)
    if (threadIdx.x == 0) {
        printf("Shared memory (logical view through swizzled layout):\n");
        for (int m = 0; m < size<0>(s_tensor); ++m) {
            printf("  row %d: ", m);
            for (int n = 0; n < size<1>(s_tensor); ++n) {
                printf("%5.0f ", s_tensor(m, n));
            }
            printf("\n");
        }
    }
}

int main()
{
    constexpr int M = 8, N = 8;

    // ─── 1. Bank conflict visualizer ───
    printf("=== Bank Conflict Visualizer ===\n\n");

    // Plain column-major layout: (8,8):(1,8)
    auto plain = Layout<Shape<_8, _8>, Stride<_1, _8>>{};

    // Swizzled layout: composition(swizzle, layout)
    //   composition applies the XOR to the flat offset that layout produces
    auto swizzled = composition(Swizzle<3, 2, 3>{}, plain);

    bank_conflict_visualizer<<<1, 1>>>(plain, "WITHOUT Swizzle");
    cudaDeviceSynchronize();

    bank_conflict_visualizer<<<1, 1>>>(swizzled, "WITH Swizzle<3,2,3>");
    cudaDeviceSynchronize();

    // ─── 2. TiledCopy with swizzled smem ───
    printf("=== TiledCopy + Swizzle ===\n\n");

    auto gmem_layout = Layout<Shape<_8, _8>, Stride<_1, _8>>{};

    // TiledCopy: 32 threads, 2 floats (64 bits) per atom
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint64_t>, float>{},   // 64 bits = 2 floats
        Layout<Shape<_4, _8>>{},                        // 32 threads in 4×8
        Layout<Shape<_1, _1>>{}
    );

    float h_data[M * N];
    for (int i = 0; i < M * N; ++i) h_data[i] = float(i);

    float* d_data;
    cudaMalloc(&d_data, sizeof(h_data));
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    int smem_bytes = M * N * sizeof(float);
    copy_with_swizzle<<<1, 32, smem_bytes>>>(
        d_data, gmem_layout, swizzled, tiled_copy);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}
```

**Expected Output:**

```text
=== Bank Conflict Visualizer ===

WITHOUT Swizzle (8 x 8):

       c0    c1    c2    c3    c4    c5    c6    c7
r0     B00   B08   B16   B24   B00   B08   B16   B24
r1     B01   B09   B17   B25   B01   B09   B17   B25
r2     B02   B10   B18   B26   B02   B10   B18   B26
r3     B03   B11   B19   B27   B03   B11   B19   B27
r4     B04   B12   B20   B28   B04   B12   B20   B28
r5     B05   B13   B21   B29   B05   B13   B21   B29
r6     B06   B14   B22   B30   B06   B14   B22   B30
r7     B07   B15   B23   B31   B07   B15   B23   B31

Row-wise bank conflicts: 8  (CONFLICTS!)

WITH Swizzle<3,2,3> (8 x 8):

       c0    c1    c2    c3    c4    c5    c6    c7
r0     B00   B08   B16   B24   B04   B12   B20   B28
r1     B01   B09   B17   B25   B05   B13   B21   B29
r2     B02   B10   B18   B26   B06   B14   B22   B30
r3     B03   B11   B19   B27   B07   B15   B23   B31
r4     B04   B12   B20   B28   B00   B08   B16   B24
r5     B05   B13   B21   B29   B01   B09   B17   B25
r6     B06   B14   B22   B30   B02   B10   B18   B26
r7     B07   B15   B23   B31   B03   B11   B19   B27

Row-wise bank conflicts: 0  (CLEAN!)

=== TiledCopy + Swizzle ===

Shared memory (logical view through swizzled layout):
  row 0:     0     8    16    24    32    40    48    56
  row 1:     1     9    17    25    33    41    49    57
  row 2:     2    10    18    26    34    42    50    58
  row 3:     3    11    19    27    35    43    51    59
  row 4:     4    12    20    28    36    44    52    60
  row 5:     5    13    21    29    37    45    53    61
  row 6:     6    14    22    30    38    46    54    62
  row 7:     7    15    23    31    39    47    55    63
```

The data is logically correct — `s_tensor(m, n)` returns the right value even though the physical addresses in shared memory have been scrambled. The swizzle is completely transparent to the reader: you access `(row, col)` the same way you always did, and CuTe routes to the swizzled address behind the scenes.

## 4. Step-by-Step Explanation

**Line: `auto plain = Layout<Shape<_8, _8>, Stride<_1, _8>>{};`**

The unswizzled column-major layout. `plain(m, n) = m * 1 + n * 8`. This is the address formula from Tutorial 01 — the same `address = coord · stride` dot product. The bank for element `(m, n)` is `(m + 8n) % 32`.

**Line: `auto swizzled = composition(Swizzle<3, 2, 3>{}, plain);`**

This is the one-line fix. `composition(f, g)` creates a new function `h(x) = f(g(x))`:

1. `plain(m, n)` converts coordinates to a flat offset: `m + 8*n`
2. `Swizzle<3, 2, 3>` XORs high bits into low bits of that offset

The result is a new layout where `swizzled(m, n)` gives a *different* flat offset — one that spreads banks evenly. CuTe stores and retrieves data through this remapped offset, so the logical view is unchanged but the physical addresses avoid conflicts.

**How the XOR works for `Swizzle<3, 2, 3>`:**

```text
Example: plain address 32 (row=0, col=4) in binary = 0b100000

  Step 1: Extract source bits [5:8] → 0b1 (bit 5 is set)
  Step 2: Shift right by S=3       → 0b100 (now at bit position 2)
  Step 3: XOR with original         → 0b100000 ^ 0b000100 = 0b100100 = 36

  Plain bank:    32 % 32 = 0   (same as col 0!)
  Swizzled bank: 36 % 32 = 4   (different — conflict eliminated)
```

The XOR takes "which region of the tile am I in?" (the high bits) and mixes it into "which bank do I hit?" (the low bits). Different regions get different scrambles, so they never collide.

**Line: `auto swizzled = composition(Swizzle<3, 2, 3>{}, plain);`** *(why these specific numbers?)*

- **M=2 (free bits):** The bottom 2 bits of the address are untouched. 2^2 = 4 consecutive floats stay contiguous = 128 bits. This preserves vectorized `STS.128` stores. If M were 0, even adjacent elements could get scrambled and vectorization would break.
- **B=3 (XOR width):** 3 bits → scramble across groups of 2^3 = 8 banks. Enough to break the 8-column pattern of our 8×8 tile.
- **S=3 (shift):** Source bits start at position M+S = 5, right above the target bits at position M = 2. No overlap between source and target.

**Line: `copy(tiled_copy, thr_g, thr_s);`**

The `TiledCopy` doesn't know about the swizzle — and doesn't need to. It partitions `s_tensor` based on its layout (which now includes the swizzle), and the `copy()` call stores through the swizzled addresses. Each thread's 2-float store still lands on contiguous addresses (because M=2 preserves 4-element contiguity), so vectorization is unaffected.

**Line: `s_tensor(m, n)` in the print loop**

Reading back through the swizzled layout is also transparent. `s_tensor(m, n)` computes the swizzled address, reads from that location, and returns the correct value. The logical view is identical to the plain layout — the scrambling only affects the physical address.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** Swizzle is **staggered brick-laying** for shared memory. In a brick wall, each row is offset so joints don't line up vertically — this prevents cracks from running straight through. In shared memory, each row's addresses are XOR-shifted so bank assignments don't repeat across columns — this prevents bank conflicts from stacking up. The `composition` call is the mortar that binds the swizzle to your layout: one line, and every access goes through the staggered pattern automatically.

**Choosing Swizzle Parameters:**

| Parameter | Rule of Thumb |
| :--- | :--- |
| **M** (free bits) | Set to `log2(vector_width / sizeof(element))`. For 128-bit loads on `float`: M = log2(128/32) = 2. For `half`: M = log2(128/16) = 3. |
| **B** (XOR width) | Set to `log2(num_columns_to_disambiguate)`. For 8 columns: B=3. For 16: B=4 (but you'll need the address space to support it). |
| **S** (shift) | Usually = B (non-overlapping source and target fields). This is the simplest and most common choice. |

**Common Swizzle Configurations in CUTLASS:**

| Swizzle | Use Case | Free Bits | Scramble Width |
| :--- | :--- | :--- | :--- |
| `Swizzle<3, 3, 3>` | 128-byte smem tiles, `half` elements | 8 halfs = 128 bits | 8 banks |
| `Swizzle<3, 2, 3>` | 128-byte smem tiles, `float` elements | 4 floats = 128 bits | 8 banks |
| `Swizzle<2, 3, 3>` | 64-byte smem tiles | 8 elements | 4 banks |
| `Swizzle<1, 3, 3>` | 32-byte smem tiles | 8 elements | 2 banks |
| `Swizzle<0, 0, 0>` | No swizzle (identity) | — | — |

**Why the swizzle doesn't break vectorization:**

The M "free" bits guarantee that blocks of 2^M consecutive elements remain at consecutive addresses after swizzling. For M=2, any group of 4 adjacent floats stays contiguous — exactly what `STS.128` needs. The swizzle only shuffles *which group of 4* goes where, not the elements within the group.

**`composition` — the key CuTe operation:**

`composition(f, g)` computes `f(g(x))`. When `f` is a `Swizzle` and `g` is a `Layout`, the result is a new layout-like object that maps coordinates to swizzled offsets. You can use it anywhere a layout is expected:

```cpp
// Unswizzled — has bank conflicts
auto smem_layout = Layout<Shape<_128, _32>, Stride<_1, _128>>{};

// Swizzled — bank-conflict-free, one-line change
auto smem_layout = composition(Swizzle<3, 2, 3>{},
                               Layout<Shape<_128, _32>, Stride<_1, _128>>{});

// Use it exactly like a normal layout
auto s_tensor = make_tensor(make_smem_ptr(smem), smem_layout);
```

**Hardware Note:** Shared memory bank conflicts show up in `ncu` (NVIDIA Nsight Compute) under the metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`. If this number is non-zero, you have conflicts. The fix is almost always a swizzle on your smem layout. On Hopper/Blackwell, all MMA swizzle modes (including no-swizzle) are bank-conflict-free on the MMA read side — the swizzle in CUTLASS's default smem layouts for WGMMA/tcgen05 is there to maximize **TMA write throughput** when populating shared memory, not to avoid read-side bank conflicts.

> **Gotcha — swizzle and `cosize`:** A swizzled layout may produce offsets larger than the plain layout's maximum. Always allocate shared memory based on `cosize(swizzled_layout)`, not `size(plain_layout)`. In practice, for well-chosen parameters (where B+M+S ≤ address bits), the max offset stays within the original range, but it's good practice to use `cosize` regardless.

> **Gotcha — debugging swizzled smem:** If you `printf` raw smem addresses, the data looks scrambled. This is expected — the physical layout *is* scrambled. Always access through the CuTe tensor (using logical coordinates), and the swizzle is transparent. If you need to dump raw smem for debugging, compose with the inverse swizzle (XOR is its own inverse — applying the same swizzle twice gives the original address).

**What comes next:** With vectorized, parallel, bank-conflict-free copies from global to shared memory, the data movement story is complete. Tutorial 06 (Hello, MMA) shifts to the *compute* side: feeding that data into a Tensor Core instruction to trigger a hardware matrix multiply.
