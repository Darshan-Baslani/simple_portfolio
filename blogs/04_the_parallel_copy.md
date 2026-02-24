# The Parallel Copy — Orchestrating Threads with TiledCopy

**Difficulty:** Beginner–Intermediate
**Prerequisites:** [Tutorial 01: Hello, Layout!](https://www.dcbaslani.xyz/blog.html?post=01_hello_layout), [Tutorial 02: The Art of Slicing](https://www.dcbaslani.xyz/blog.html?post=02_the_art_of_slicing), [Tutorial 03: The Naive Copy](https://www.dcbaslani.xyz/blog.html?post=03_the_naive_copy)

## 1. The Problem (The "Why")

Tutorial 03 showed how one thread can vectorize its copy — using a dolly instead of hand-carrying boxes. But a real kernel doesn't have one worker on the loading dock; it has 32, 128, or 256 workers that all need to copy the *same tile together*, without stepping on each other's toes.

You could combine Tutorial 02's `local_partition` with Tutorial 03's `copy()`:

```cpp
// Manual approach — works, but fragile
auto thr_g = local_partition(g_tensor, Layout<Shape<_8, _32>>{}, threadIdx.x);
auto thr_s = local_partition(s_tensor, Layout<Shape<_8, _32>>{}, threadIdx.x);
copy(AutoVectorizingCopy{}, thr_g, thr_s);
```

This works — but *you* have to manually choose a thread layout, make sure it divides the tile evenly, and hope the resulting per-thread partition is contiguous enough for vectorization. Change the tile size? Redo the math. Transpose the layout? Redo it again. Switch from `LDG` to `cp.async`? Rewrite the whole thing.

CuTe's `TiledCopy` bundles all three decisions into one declarative object:

| Piece | What it controls | From Tutorial... |
| :--- | :--- | :--- |
| `Copy_Atom` | What each thread can carry in one trip (the dolly) | 03 |
| `thr_layout` | Where each thread stands on the tile (their grid position) | 02 |
| `val_layout` | How many values each thread handles per trip | New |

You declare these once. CuTe generates the partitioning, the vectorization, and the thread-to-element mapping automatically.

> **B200 Note:** On Hopper/Blackwell, `TiledCopy` is also the container for TMA and `cp.async` atoms. The `thr_layout`/`val_layout` mechanism is identical — only the `Copy_Atom` changes. Master this pattern now and TMA (Tutorial 08) becomes a one-line swap.

## 2. The Mental Model (The Visual)

Imagine you have a **16×8 tile** to copy and **32 threads** (one warp). You need to answer three questions:

1. **What tool does each thread use?** → `Copy_Atom` with `UniversalCopy<uint128_t>` — each thread moves 128 bits (4 floats) per load instruction. This is the dolly from Tutorial 03.
2. **How are threads arranged?** → `thr_layout = (4, 8)` — 4 rows of threads, 8 columns.
3. **How many extra values per thread?** → `val_layout = (1, 1)` — the atom's 4 floats are enough; no extra tiling needed.

```text
Tile coverage = thr_layout × atom_values × val_layout
              = (4, 8)    × (4, 1)      × (1, 1)
              = (16, 8)   ✓ matches our tile!
```

### The Ownership Map

Each cell shows which thread (T00–T31) copies it. The atom's 4 floats go along mode-0 (down the column), so each thread owns a 4×1 vertical strip:

```text
              col 0    col 1    col 2    col 3    col 4    col 5    col 6    col 7
            ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
   row  0   │  T00   │  T04   │  T08   │  T12   │  T16   │  T20   │  T24   │  T28   │
   row  1   │  T00   │  T04   │  T08   │  T12   │  T16   │  T20   │  T24   │  T28   │ ← T00 owns
   row  2   │  T00   │  T04   │  T08   │  T12   │  T16   │  T20   │  T24   │  T28   │   rows 0–3,
   row  3   │  T00   │  T04   │  T08   │  T12   │  T16   │  T20   │  T24   │  T28   │   col 0
            ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
   row  4   │  T01   │  T05   │  T09   │  T13   │  T17   │  T21   │  T25   │  T29   │
   row  5   │  T01   │  T05   │  T09   │  T13   │  T17   │  T21   │  T25   │  T29   │
   row  6   │  T01   │  T05   │  T09   │  T13   │  T17   │  T21   │  T25   │  T29   │
   row  7   │  T01   │  T05   │  T09   │  T13   │  T17   │  T21   │  T25   │  T29   │
            ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
   row  8   │  T02   │  T06   │  T10   │  T14   │  T18   │  T22   │  T26   │  T30   │
   row  9   │  T02   │  T06   │  T10   │  T14   │  T18   │  T22   │  T26   │  T30   │
   row 10   │  T02   │  T06   │  T10   │  T14   │  T18   │  T22   │  T26   │  T30   │
   row 11   │  T02   │  T06   │  T10   │  T14   │  T18   │  T22   │  T26   │  T30   │
            ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
   row 12   │  T03   │  T07   │  T11   │  T15   │  T19   │  T23   │  T27   │  T31   │
   row 13   │  T03   │  T07   │  T11   │  T15   │  T19   │  T23   │  T27   │  T31   │
   row 14   │  T03   │  T07   │  T11   │  T15   │  T19   │  T23   │  T27   │  T31   │
   row 15   │  T03   │  T07   │  T11   │  T15   │  T19   │  T23   │  T27   │  T31   │
            └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

Notice the two critical patterns:

- **Within each thread:** T00's 4 values (rows 0–3 of col 0) are contiguous in column-major memory. That's 4 floats = 128 bits — a single `LDG.128`. The atom's `uint128_t` type made this happen.
- **Across threads:** Each column has a different thread (T00, T04, T08, ...). No two threads touch the same cell. The `thr_layout = (4,8)` made this happen.

### The Three-Part Recipe

```text
┌──────────────────────────────────┐
│         Copy_Atom                │ ← The tool: "what can one thread carry?"
│  (UniversalCopy<uint128_t>,      │    Each load moves 4 floats / 128 bits.
│   float)                         │    (the dolly from Tutorial 03)
└────────────────┬─────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───┴──────────┐   ┌─────────┴──────────┐
│  thr_layout  │   │    val_layout       │
│  (4, 8)      │   │    (1, 1)           │
│              │   │                     │
│  "32 workers │   │  "no extra values   │
│   in a 4×8   │   │   beyond what the   │
│   grid"      │   │   atom handles"     │
└──────────────┘   └────────────────────┘

Tile covered = (4 thr × 4 atom × 1 val, 8 thr × 1 atom × 1 val) = (16, 8)
```

> **The Moving Crew Rule:** Tutorial 03 was about one mover and their dolly. `TiledCopy` is the **shift supervisor's clipboard** — a plan that assigns every mover to a grid position on the warehouse floor and specifies what tool to use. The `Copy_Atom` says what each mover carries (the dolly). The `thr_layout` says where each mover stands. The `val_layout` says how many extra trips each mover makes beyond one atom load. Together, these three things tile the entire room with no gaps and no overlaps.

## 3. The Solution (The Code)

Two kernels: one visualizes the thread-to-element ownership map, one performs an actual global → shared copy.

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>

#include <cstdio>

using namespace cute;

// ─── Kernel 1: Visualize which thread owns which tile cell ───
template <class TiledCopy, class TileLayout>
__global__ void visualize_ownership(TiledCopy tiled_copy, TileLayout tile_layout)
{
    extern __shared__ float smem[];

    // Build a shared-memory tensor to stamp thread IDs into
    auto tile = make_tensor(make_smem_ptr(smem), tile_layout);

    // Get this thread's copy slice
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_view = thr_copy.partition_D(tile);

    // Each thread writes its ID into the cells it owns
    for (int i = 0; i < size(thr_view); ++i) {
        thr_view(i) = float(threadIdx.x);
    }
    __syncthreads();

    // Thread 0 prints the full ownership map
    if (threadIdx.x == 0) {
        printf("Tile ownership map (%d x %d), %d threads:\n\n",
               int(size<0>(tile)), int(size<1>(tile)), int(blockDim.x));

        printf("       ");
        for (int n = 0; n < size<1>(tile); ++n) printf("c%-4d ", n);
        printf("\n");

        for (int m = 0; m < size<0>(tile); ++m) {
            printf("r%-4d  ", m);
            for (int n = 0; n < size<1>(tile); ++n) {
                printf("T%02d   ", int(tile(m, n)));
            }
            printf("\n");
        }
    }
}

// ─── Kernel 2: Actual Global → Shared copy using TiledCopy ───
template <class TiledCopy, class GmemLayout, class SmemLayout>
__global__ void tiled_copy_kernel(float const* __restrict__ g_ptr,
                                  GmemLayout   gmem_layout,
                                  SmemLayout   smem_layout,
                                  TiledCopy    tiled_copy)
{
    extern __shared__ float smem[];

    auto g_tensor = make_tensor(make_gmem_ptr(g_ptr), gmem_layout);  // (M, N)
    auto s_tensor = make_tensor(make_smem_ptr(smem),  smem_layout);  // (M, N)

    // ── The key lines: partition via TiledCopy ──
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_g    = thr_copy.partition_S(g_tensor);    // source partition
    auto thr_s    = thr_copy.partition_D(s_tensor);    // destination partition

    // ── Execute the copy ──
    copy(tiled_copy, thr_g, thr_s);

    __syncthreads();

    // Thread 0 prints a few rows to verify correctness
    if (threadIdx.x == 0) {
        printf("\nShared memory after TiledCopy (column-major, showing row by row):\n");
        for (int m = 0; m < size<0>(s_tensor); ++m) {
            printf("  row %2d: ", m);
            for (int n = 0; n < size<1>(s_tensor); ++n) {
                printf("%5.0f ", s_tensor(m, n));
            }
            printf("\n");
        }
    }
}

int main()
{
    constexpr int M = 16, N = 8;

    // ─── Build the TiledCopy ───
    // 1. Copy_Atom: each load moves 128 bits (4 floats) via uint128_t
    // 2. thr_layout: 32 threads in a 4×8 grid (column-major)
    // 3. val_layout: 1×1 — the atom already handles 4 floats per thread
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        Layout<Shape<_4, _8>>{},      // thr_layout: 4 rows × 8 cols = 32 threads
        Layout<Shape<_1, _1>>{}       // val_layout: atom covers everything
    );

    // Tile layouts: column-major for both
    auto tile_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutLeft{});

    int smem_bytes = M * N * sizeof(float);

    // ─── Kernel 1: Visualize the ownership map ───
    printf("=== TiledCopy Ownership Map ===\n\n");
    visualize_ownership<<<1, 32, smem_bytes>>>(tiled_copy, tile_layout);
    cudaDeviceSynchronize();

    // ─── Kernel 2: Actual Global → Shared copy ───
    // Fill source: h_data[i] = i (flat index in column-major order)
    float h_data[M * N];
    for (int i = 0; i < M * N; ++i) h_data[i] = float(i);

    float* d_data;
    cudaMalloc(&d_data, sizeof(h_data));
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    printf("\n=== TiledCopy: Global -> Shared ===\n");
    tiled_copy_kernel<<<1, 32, smem_bytes>>>(d_data, tile_layout, tile_layout, tiled_copy);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}
```

**Expected Output:**

```text
=== TiledCopy Ownership Map ===

Tile ownership map (16 x 8), 32 threads:

       c0    c1    c2    c3    c4    c5    c6    c7
r0     T00   T04   T08   T12   T16   T20   T24   T28
r1     T00   T04   T08   T12   T16   T20   T24   T28
r2     T00   T04   T08   T12   T16   T20   T24   T28
r3     T00   T04   T08   T12   T16   T20   T24   T28
r4     T01   T05   T09   T13   T17   T21   T25   T29
r5     T01   T05   T09   T13   T17   T21   T25   T29
r6     T01   T05   T09   T13   T17   T21   T25   T29
r7     T01   T05   T09   T13   T17   T21   T25   T29
r8     T02   T06   T10   T14   T18   T22   T26   T30
r9     T02   T06   T10   T14   T18   T22   T26   T30
r10    T02   T06   T10   T14   T18   T22   T26   T30
r11    T02   T06   T10   T14   T18   T22   T26   T30
r12    T03   T07   T11   T15   T19   T23   T27   T31
r13    T03   T07   T11   T15   T19   T23   T27   T31
r14    T03   T07   T11   T15   T19   T23   T27   T31
r15    T03   T07   T11   T15   T19   T23   T27   T31

=== TiledCopy: Global -> Shared ===

Shared memory after TiledCopy (column-major, showing row by row):
  row  0:     0    16    32    48    64    80    96   112
  row  1:     1    17    33    49    65    81    97   113
  row  2:     2    18    34    50    66    82    98   114
  row  3:     3    19    35    51    67    83    99   115
  row  4:     4    20    36    52    68    84   100   116
  row  5:     5    21    37    53    69    85   101   117
  row  6:     6    22    38    54    70    86   102   118
  row  7:     7    23    39    55    71    87   103   119
  row  8:     8    24    40    56    72    88   104   120
  row  9:     9    25    41    57    73    89   105   121
  row 10:    10    26    42    58    74    90   106   122
  row 11:    11    27    43    59    75    91   107   123
  row 12:    12    28    44    60    76    92   108   124
  row 13:    13    29    45    61    77    93   109   125
  row 14:    14    30    46    62    78    94   110   126
  row 15:    15    31    47    63    79    95   111   127
```

The values confirm column-major order: element 0 at (0,0), element 1 at (1,0), ..., element 16 at (0,1). The `TiledCopy` moved every element to the right shared memory cell — 32 threads, zero conflicts, all vectorized.

## 4. Step-by-Step Explanation

**Line: `auto tiled_copy = make_tiled_copy(Copy_Atom{...}, thr_layout, val_layout);`**

This is the declaration — the shift supervisor's clipboard. Three ingredients:

1. **`Copy_Atom<UniversalCopy<uint128_t>, float>{}`** — The copy atom. `uint128_t` means each thread loads 128 bits = 4 floats in a single instruction (`LDG.128`). The `float` tells CuTe the logical element type.

2. **`Layout<Shape<_4, _8>>{}`** — Thread layout. 32 threads arranged as 4 rows × 8 columns in column-major order. Thread 0 → position (0,0), Thread 1 → (1,0), ..., Thread 4 → (0,1), ..., Thread 31 → (3,7).

3. **`Layout<Shape<_1, _1>>{}`** — Value layout. No extra values per thread beyond the atom's 4 floats. If the tile were bigger, you'd increase this to cover the extra elements (see "Bigger Tiles" below).

**Tile coverage:** `(4 threads × 4 atom_vals × 1 val, 8 threads × 1 atom_val × 1 val) = (16, 8)`.

**Line: `auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);`**

Each thread asks the `TiledCopy` for its personal *slice* — a lightweight object that knows which tile cells belong to this thread. This is each mover reading their assignment off the clipboard.

**Line: `auto thr_g = thr_copy.partition_S(g_tensor);`**

Partitions the source (global memory) tensor for this thread. The result is a tensor containing only the elements this thread should read. The shape is `(ValuesPerAtom, Repetitions...)`:

- For our 16×8 tile: shape `(_4)` — four contiguous floats loaded as one `uint128_t`. No repetitions because the tile matches the `TiledCopy` coverage exactly.
- If the tile were **32×8**: shape `(_4, _2)` — four floats per load × 2 rounds along mode-0. CuTe adds the repetition dimension automatically.

**Line: `auto thr_s = thr_copy.partition_D(s_tensor);`**

Same for the destination (shared memory). Its shape matches `thr_g` so that `copy()` knows where each source value lands.

**Line: `copy(tiled_copy, thr_g, thr_s);`**

Executes the copy. CuTe loops over all repetition dimensions (if any), applying the `Copy_Atom` to each batch of values. For our 16×8 tile:

- 32 threads × 4 floats each = 128 floats = entire tile
- Each thread issues one `LDG.128` (global load) and one `STS.128` (shared store)
- Total: 32 loads + 32 stores, all vectorized — the entire tile moves in one round

### Bigger Tiles and Repetitions

What if your tile is 128×128 but you still have 32 threads?

```cpp
auto big_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, float>{},
    Layout<Shape<_4, _8>>{},      // still 32 threads
    Layout<Shape<_1, _1>>{}       // same atom
);
// TiledCopy covers (16, 8) per round.
// Tile is (128, 128) → repetitions = (128/16, 128/8) = (8, 16).
// partition_S returns shape (_4, _8, _16).
// copy() loops over the 8×16 = 128 repetitions automatically.
```

Or you can increase `val_layout` to give each thread more work per round:

```cpp
auto wide_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, float>{},
    Layout<Shape<_4, _8>>{},      // 32 threads
    Layout<Shape<_2, _1>>{}       // each thread handles 2 columns of atom-loads
);
// Coverage per round: (4×4×2, 8×1×1) = (32, 8).
// partition_S for a 128×128 tile: shape (_4, _2, _4, _16).
```

Don't worry about memorizing these shapes — the point is that `copy()` handles the loops. You just set up the `TiledCopy` and call `copy()`.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** Tutorial 03 was about one mover and their dolly. `TiledCopy` is the **shift supervisor's clipboard** — a plan that assigns every mover to a grid position on the warehouse floor. The `Copy_Atom` says what tool each mover uses (the dolly). The `thr_layout` is where they stand. The `val_layout` is how many extra trips each mover makes. Three pieces, one plan, zero overlap.

**The `val_layout` and Vectorization:**

The atom's vector direction must align with the contiguous memory dimension, or you lose vectorization:

| Memory Layout | Atom Type | Why It Works |
| :--- | :--- | :--- |
| Column-major `(M,N):(1,M)` | `UniversalCopy<uint128_t>` with thr (4,8) | 4 floats along mode-0 are stride-1 → `LDG.128` |
| Row-major `(M,N):(N,1)` | `UniversalCopy<uint128_t>` with thr (8,4) | Swap thread arrangement so the atom's 4 values hit stride-1 mode-1 |
| Strided / dynamic | `UniversalCopy<float>` | Scalar fallback — always works, always slow |

If the atom's 4-element chunk lands on non-contiguous addresses, the hardware can't coalesce them and you're back to scalar speed. Always check: *"which dimension has stride 1?"* — and point the atom's vector width in that direction.

**The `TiledCopy` API Cheat-Sheet:**

| API Call | What It Does |
| :--- | :--- |
| `make_tiled_copy(atom, thr, val)` | Build the plan from three ingredients |
| `tiled_copy.get_thread_slice(tid)` | Each worker reads their assignment |
| `slice.partition_S(src)` | Partition source for this thread |
| `slice.partition_D(dst)` | Partition destination for this thread |
| `copy(tiled_copy, thr_src, thr_dst)` | Execute the copy (loops over repetitions) |

**`TiledCopy` vs. Manual `local_partition` + `copy`:**

| | `local_partition` + `copy` | `TiledCopy` |
| :--- | :--- | :--- |
| Thread mapping | You compute it | Declared once |
| Vectorization | Depends on partition contiguity | Follows from atom type |
| Swap to `cp.async` / TMA | Rewrite everything | Change one `Copy_Atom` |
| Repetitions for big tiles | Manual loop | Automatic |

**Hardware Note:** For global memory loads, the GPU coalesces requests from threads in the same warp that hit the same 128-byte cache line. To maximize coalescing, threads with adjacent IDs should access adjacent memory addresses. For column-major data, spread threads along mode-0 (rows) first — which is what `thr_layout = (T_rows, T_cols)` in column-major order does naturally: threads 0, 1, 2, 3 hit rows 0–3 of the same column, which are adjacent in memory.

> **Gotcha — partition shape surprise:** `partition_S` and `partition_D` return tensors shaped `(ValuesPerAtom, Repetitions...)`, NOT `(M, N)`. The first mode is the "hand" for a single atom invocation; the remaining modes are how many times the atom repeats to cover the tile. If you `print()` the partition and see an unexpected number of modes, it's because your tile is bigger than the `TiledCopy`'s single-round coverage. This is normal — `copy()` iterates over the repetitions automatically.

**What comes next:** Each thread's copy is now vectorized and coordinated. But when 32 threads simultaneously write to shared memory, they can collide on the *same bank*. If threads 0, 8, 16, 24 all write to addresses that map to shared memory bank 0, the hardware serializes those writes — a **bank conflict**. That's the topic of Tutorial 05 (Swizzling), where `composition` with a `Swizzle` layout remaps addresses to spread them across banks.
