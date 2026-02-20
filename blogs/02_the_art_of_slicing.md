# The Art of Slicing — Partitioning Data Across Blocks and Threads

**Difficulty:** Beginner
**Prerequisites:** [Tutorial 01: Hello, Layout!](01_hello_layout.md), CUDA thread blocks (`blockIdx`, `threadIdx`)

## 1. The Problem (The "Why")

You have a big matrix — say 512×512 — sitting in global memory. Your GPU launches a grid of thread blocks, each of which should work on a small tile (e.g., 128×128). Inside each block, 256 threads need to split *that* tile further so every thread handles its own little piece.

Without CuTe, you write something like:

```cpp
int row_start = blockIdx.x * TILE_M;
int col_start = blockIdx.y * TILE_N;
int thr_row   = threadIdx.x % THR_M;
int thr_col   = threadIdx.x / THR_M;
// ... and then more math for every matrix, every stage, every layout change
```

This is fragile. Change the tile size? Rewrite the math. Transpose a matrix? Rewrite it again. Add a K-dimension for GEMM? Rewrite it *again*.

CuTe replaces all of this with two composable operations:

| Operation | Level | What it does |
| :--- | :--- | :--- |
| `local_tile` | Block (CTA) | "Give this block its tile from the big matrix." |
| `local_partition` | Thread | "Give this thread its elements from the tile." |

Both work on any shape, any stride, any number of dimensions.

> **B200 Note:** When you program the TMA engine on Hopper/Blackwell, you still need `local_tile` to select *which* tile the TMA should fetch. The TMA descriptor handles the copy, but you tell it *where* to point using the exact same tiling logic described here.

## 2. The Mental Model (The Visual)

### Step 1: The Big Picture — Tiling for CTAs

Imagine an 8×8 matrix divided into 4×4 tiles. Four thread blocks share the work:

```text
                       Global Matrix (8×8)
         col 0  col 1  col 2  col 3  col 4  col 5  col 6  col 7
       ┌───────────────────────────┬───────────────────────────┐
row 0  │                           │                           │
row 1  │      Block (0,0)          │      Block (0,1)          │
row 2  │      rows 0-3, cols 0-3   │      rows 0-3, cols 4-7   │
row 3  │                           │                           │
       ├───────────────────────────┼───────────────────────────┤
row 4  │                           │                           │
row 5  │      Block (1,0)          │      Block (1,1)          │
row 6  │      rows 4-7, cols 0-3   │      rows 4-7, cols 4-7   │
row 7  │                           │                           │
       └───────────────────────────┴───────────────────────────┘

  local_tile(matrix, Shape<4,4>{}, make_coord(blockIdx.x, blockIdx.y))
  → extracts one 4×4 tile for each block
```

`local_tile` applies a **tiler** (here `Shape<4,4>`) to produce tiles, then uses the block coordinate to pick the right one. Under the hood it calls `zipped_divide`, which reshapes the matrix into `((Tile), (Rest))` — tile elements in the first mode, tile indices in the second — then slices the rest-mode with your coordinate.

### Step 2: Inside a Tile — Partitioning for Threads

Now zoom into Block (0,0)'s 4×4 tile. We have 4 threads arranged as a 2×2 layout. Each thread gets a 2×2 chunk:

```text
            Block (0,0)'s Tile (4×4)

         col 0   col 1   col 2   col 3
       ┌─────────────────┬─────────────────┐
row 0  │  T0      T0     │  T1      T1     │
row 1  │  T0      T0     │  T1      T1     │
       ├─────────────────┼─────────────────┤
row 2  │  T2      T2     │  T3      T3     │
row 3  │  T2      T2     │  T3      T3     │
       └─────────────────┴─────────────────┘

  Thread layout: (2,2):(1,2)  →  4 threads in column-major
  Tile shape:    (4,4)
  Each thread:   (4/2, 4/2) = (2,2) elements

  local_partition(tile, Layout<Shape<2,2>>{}, threadIdx.x)
  → Thread 0 gets the top-left 2×2 subtensor
```

`local_partition` takes a **thread layout** — a Layout that maps coordinates to thread indices — and uses `zipped_divide` to split the tile into thread-sized pieces, then slices into the tile-mode with the thread's *coordinate* (derived from its flat index via the thread layout).

> **Key Insight:** `local_tile` slices the **rest** (picks which tile). `local_partition` slices the **tile** (picks which thread's elements). They are two sides of the same `zipped_divide` coin.

## 3. The Solution (The Code)

A CUDA kernel that tiles a matrix across blocks, partitions each tile across threads, and prints what each agent owns.

```cpp
#include <cute/tensor.hpp>

using namespace cute;

template <class Shape_MN, class CtaTiler, class ThreadLayout>
__global__ void partition_kernel(Shape_MN shape_mn,
                                 CtaTiler cta_tiler,
                                 ThreadLayout thr_layout)
{
    // 1. Build the full matrix as a counting tensor (value at (i,j) = flat index)
    //    In real code this would be make_tensor(make_gmem_ptr(ptr), shape, stride)
    auto matrix = make_counting_tensor(make_layout(shape_mn));          // (M,N)

    // 2. CTA-level: each block picks its tile
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
    auto cta_tile  = local_tile(matrix, cta_tiler, cta_coord);         // (BLK_M,BLK_N)

    // 3. Thread-level: each thread picks its elements from the tile
    auto thr_tile  = local_partition(cta_tile, thr_layout, threadIdx.x); // (THR_M,THR_N)

    // 4. Print from Block (0,0), Thread 0 only
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("=== Full matrix layout ===\n");
        print(matrix.layout()); printf("\n\n");

        printf("=== Block (0,0) tile ===\n");
        print(cta_tile.layout()); printf("\n");
        for (int m = 0; m < size<0>(cta_tile); ++m) {
            for (int n = 0; n < size<1>(cta_tile); ++n) {
                printf("%3d ", int(cta_tile(m, n)));
            }
            printf("\n");
        }

        printf("\n=== Block (0,0), Thread 0 partition ===\n");
        print(thr_tile.layout()); printf("\n");
        printf("Thread 0 owns elements: ");
        for (int i = 0; i < size(thr_tile); ++i) {
            printf("%d ", int(thr_tile(i)));
        }
        printf("\n");
    }
}

int main()
{
    // 8×8 matrix, tiled into 4×4 blocks, 4 threads per block (2×2)
    auto shape_mn   = make_shape(Int<8>{}, Int<8>{});
    auto cta_tiler  = make_shape(Int<4>{}, Int<4>{});
    auto thr_layout = make_layout(make_shape(Int<2>{}, Int<2>{}));     // (m,n)->thr_idx

    dim3 grid(size(ceil_div(get<0>(shape_mn), get<0>(cta_tiler))),
              size(ceil_div(get<1>(shape_mn), get<1>(cta_tiler))));    // 2×2 blocks
    dim3 block(size(thr_layout));                                      // 4 threads

    partition_kernel<<<grid, block>>>(shape_mn, cta_tiler, thr_layout);
    cudaDeviceSynchronize();
    return 0;
}
```

**Expected Output:**

```text
=== Full matrix layout ===
(_8,_8):(_1,_8)

=== Block (0,0) tile ===
(_4,_4):(_1,_8)
  0   8  16  24
  1   9  17  25
  2  10  18  26
  3  11  19  27

=== Block (0,0), Thread 0 partition ===
(_2,_2):(_1,_8)
Thread 0 owns elements: 0 1 8 9
```

Block (0,0) gets the top-left 4×4 tile (addresses 0–3 down column 0, 8–11 down column 1, etc. — column-major). Thread 0 gets the top-left 2×2 corner of that tile.

## 4. Step-by-Step Explanation

**Line: `auto matrix = make_counting_tensor(make_layout(shape_mn));`**

- `make_counting_tensor` creates a "virtual" tensor where the value at each coordinate equals its flat index. No actual memory — it's a trick for visualizing the Layout's mapping. In a real kernel, replace this with `make_tensor(make_gmem_ptr(ptr), shape, stride)`.

**Line: `auto cta_tile = local_tile(matrix, cta_tiler, cta_coord);`**

- `local_tile` is CuTe's **inner partition**. Under the hood:
  1. `zipped_divide(matrix, cta_tiler)` reshapes the `(8,8)` matrix into `((4,4),(2,2))` — a 4×4 tile mode and a 2×2 "rest" mode (2 tiles per row, 2 tiles per column).
  2. The `cta_coord` slices the rest-mode: `(blockIdx.x, blockIdx.y)` picks one tile.
- Block (0,0) gets the tile at rest-coordinate (0,0) — the top-left 4×4 region.

**Line: `auto thr_tile = local_partition(cta_tile, thr_layout, threadIdx.x);`**

- `local_partition` is CuTe's **outer partition**. Under the hood:
  1. It takes the *shape* of `thr_layout` — here `(2,2)` — and uses it as a tiler on the `(4,4)` tile, producing `((2,2),(2,2))` — 2×2 elements per thread, tiled over 2×2 threads.
  2. It converts `threadIdx.x` into a coordinate using `thr_layout`. Thread 0 → coord `(0,0)`, Thread 1 → `(1,0)`, Thread 2 → `(0,1)`, Thread 3 → `(1,1)`.
  3. That coordinate slices the tile-mode, so each thread gets the matching 2×2 block of elements.

**Line: `ceil_div(get<0>(shape_mn), get<0>(cta_tiler))`**

- `ceil_div(8, 4) = 2`. This gives us 2 blocks in the M-dimension (and 2 in N), for a total of 4 blocks.

**`cosize` (used in real kernels for shared memory):**

- `cosize(layout)` returns the size of the layout's codomain — the number of unique addresses needed. Use it to allocate shared memory: `__shared__ float smem[cosize_v<SmemLayout>];`.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** Think of partitioning as **dealing cards at a poker table**:

1. **`local_tile`** = The casino floor manager assigns a *table* (tile) to each *dealer* (CTA). "Table 0, you get the top-left quarter of the deck."
2. **`local_partition`** = The dealer at each table divides the cards among the *players* (threads). "Player 0, you get these 4 cards."
3. **`zipped_divide`** = The reorganization step that makes dealing possible. It splits the deck into `((hand), (tables))` so you can ask for any specific table's hand in one shot.

The key invariant: **every element is owned by exactly one block and exactly one thread within that block.** No overlap, no gaps (assuming tile sizes evenly divide the matrix — see the [predication tutorial](0y_predication.md) for handling remainders).

**Hardware Note:** On Hopper/B200, the tile coordinate you compute with `local_tile` is the *same* coordinate you pass to `cute::copy` with a TMA atom. The TMA engine uses the descriptor's shape/stride (from Tutorial 01) and this coordinate to DMA the right tile into shared memory — zero address math in registers.

**The `local_tile` / `local_partition` Cheat-Sheet:**

| Function | Level | Slices into... | You provide... | You get... |
| :--- | :--- | :--- | :--- | :--- |
| `local_tile` | CTA | Rest (which tile) | Tile shape + block coord | The tile data |
| `local_partition` | Thread | Tile (which thread) | Thread layout + thread idx | That thread's elements |

**Under the hood — both use `zipped_divide`:**

```text
zipped_divide(data, tiler)  →  ((Tile), (Rest))
                                  │         │
                         local_partition  local_tile
                         slices here      slices here
```
