# Cute-DSL: I Wrote a CUDA Kernel in Python and My GPU Didn't Even Cry

Welcome to the ultimate guide to `cute-dsl`! If you've tinkered with CUDA programming and are curious about CUTLASS's CuTe but felt overwhelmed by the complex C++ templates, `cute-dsl` is your new best friend. It brings the power of CuTe's concepts—like Layouts, Tilers, and vectorized memory operations—straight into a familiar, Pythonic interface.

In this blog, we'll dive into the basics, starting with simple APIs, building intuition around memory layouts with ASCII diagrams, and graduating to vectorized kernels.

## The Essentials: `cute.kernel` and `cute.jit`

At the core of `cute-dsl` are two crucial decorators:
- `@cute.kernel`: This marks a Python function as a device kernel, meaning it will run natively on the GPU. It behaves similarly to PyTorch's Triton `@triton.jit` or Numba's `@cuda.jit`.
- `@cute.jit`: This marks a host function that will compile and launch your device kernels. Think of it as the bridge between your standard Python code and your GPU parallel executions.

### A Gentle Start: Hello, GPU
Let's look at the absolute simplest program you can write:

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def hello_kernel():
    # Fetch thread coordinates just like CUDA!
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("hello from gpu")

@cute.jit
def hello_world():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))

if __name__ == "__main__":
    compiled = cute.compile(hello_world)
    compiled()
```

When you define `hello_kernel`, it gets compiled into PTX code behind the scenes. Inside it, we use `cute.arch.thread_idx()` — which returns `(threadIdx.x, threadIdx.y, threadIdx.z)`, just like in normal CUDA. We then launch it using standard `(grid, block)` dimensions inside a `@cute.jit`-compiled host function. Beautifully simple!

## Demystifying Layouts

A **Layout** in `cute` is the fundamental building block of memory access. It dictates how a logical multi-dimensional coordinate maps to a 1D flat memory index.

A Layout is parameterized by two things: **Shape** and **Stride**.

Let's imagine an 8x8 matrix in **Column-Major** layout: `shape=(8, 8)` and `stride=(1, 8)`.

The formula to convert a multi-dimensional coordinate to a flat memory index is: `index = row * stride[0] + col * stride[1]`. With stride `(1, 8)`, moving one step down a column (incrementing row) moves by 1 in memory, while moving one step across a row (incrementing column) jumps by 8.

```text
Logical 8x8 View (showing flat index)       Flat Memory (Column-Major)
+----+----+----+-----                       -----------
|  0 |  8 | 16 | ...                        [0] -> (row=0, col=0)
+----+----+----+-----                       [1] -> (row=1, col=0)
|  1 |  9 | 17 | ...                        [2] -> (row=2, col=0)
+----+----+----+-----                       ...
|  2 | 10 | 18 | ...                        [8] -> (row=0, col=1)
+----+----+----+-----                       [9] -> (row=1, col=1)
```

In **Row-Major** layout, the stride would be `(8, 1)` — meaning moving to the *next row* jumps by 8 in flat memory, while moving to the *next column* steps by just 1. So elements within the same row are contiguous in memory, which is the opposite of column-major.

Layouts in `cute` can also be **hierarchically nested**. For example, a shape of `(2, (2, 4))` with stride `(1, (2, 4))` describes a 2-element outer dimension where each element contains a nested 2×4 sub-layout. This nesting lets you mathematically model complex memory access patterns like blocked or swizzled layouts — essential for high-performance GPU kernels.

## Logical Divide vs Zipped Divide

When we want to transfer data between different memory hierarchies (like Global Memory -> Shared Memory -> Registers), we often do so in small parallelized blocks called **Tiles**. 

Let's take our 8x8 matrix layout `L = (8, 8)` and divide it using a **Tiler** of size `(2, 4)`. A tiler dictates the shape of the blocks we want to chop our layout into.

```python
L = cute.make_layout(shape=(8, 8), stride=(1, 8))
tiler = (2, 4)

L_logical = cute.logical_divide(L, tiler)
L_zipped = cute.zipped_divide(L, tiler)
```

### Logical Divide
A **Logical Divide** splits each mode independently into `(tile_size, num_tiles)`:
- Rows (size 8) ÷ Tiler Row (2) → `(2, 4)` meaning "2 rows per tile, 4 tile-rows"
- Cols (size 8) ÷ Tiler Col (4) → `(4, 2)` meaning "4 cols per tile, 2 tile-cols"

The combined output shape becomes: `((2, 4), (4, 2))`.

```text
Logical Divide: ((2, 4), (4, 2))

   Mode 0 (Rows):  (row_within_tile, which_tile_row)  =  (2, 4)
   Mode 1 (Cols):  (col_within_tile, which_tile_col)  =  (4, 2)

Indexing requires specifying all 4 sub-dimensions:
   Data[ (row_in_tile, tile_row_idx), (col_in_tile, tile_col_idx) ]

The row and column modes remain separate — they are NOT grouped
by "tile" vs "grid". This makes it awkward to grab a full 2x4 tile.
```

### Zipped Divide
A **Zipped Divide** first performs a logical divide, then *re-groups* the results: it bundles all the "within-tile" dimensions into Mode 0 and all the "which-tile" dimensions into Mode 1.
- **Mode 0 (Tile)**: `(2, 4)` — the shape of one tile (rows-in-tile, cols-in-tile)
- **Mode 1 (Grid)**: `(4, 2)` — how many tiles (tile-rows, tile-cols)

The numbers look identical to logical divide — `((2, 4), (4, 2))` — but the *meaning* of each mode has changed completely!

```text
Zipped Divide: Original (8, 8) ÷ Tiler (2, 4)

Mode 0 = One Tile (2, 4):            Mode 1 = Grid of Tiles (4, 2):

  col 0  col 1  col 2  col 3           tile_row  tile_col
  +------+------+------+------+        +-------+-------+
  |      |      |      |      |  row0  | (0,0) | (0,1) |
  +------+------+------+------+        +-------+-------+
  |      |      |      |      |  row1  | (1,0) | (1,1) |
  +------+------+------+------+        +-------+-------+
  A single 2x4 tile                    | (2,0) | (2,1) |
                                       +-------+-------+
                                       | (3,0) | (3,1) |
                                       +-------+-------+
                                       4×2 grid = 8 tiles total
```

### Why is Zipped Divide Preferred?
Zipped divide gives you a clean two-level hierarchy: `(Tile_Coordinate, Grid_Coordinate)`.

When doing vectorized loads or memory copies, you typically want to grab a *whole tile* at a specific *grid position*. Compare the two approaches:

- **Logical Divide**: To grab a tile you must index 4 sub-dimensions separately: `Data[(row_in_tile, tile_row_idx), (col_in_tile, tile_col_idx)]` — cumbersome and error-prone.
- **Zipped Divide**: The layout already mirrors `(Tile, Grid)`, so you cleanly index: `Data[tile_coord, grid_coord]` — much simpler!

This is why `zipped_divide` is the go-to operation for partitioning data across threads and memory hierarchies in CuTe kernels.

## Anatomy of the Naive Elementwise Add Kernel

Let's see how indexing looks across different paradigms for a simple elementwise `C = A + B` kernel.

```python
@cute.kernel
def naive_elementwise_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    
    m, n = A.shape
    ni = idx % n
    mi = idx // n

    C[mi, ni] = A[mi, ni] + B[mi, ni]
```

### The Indexing Difference
- **PyTorch**: Fully abstracted — you simply write `C = A + B` and PyTorch handles parallelism, memory layout, and iteration internally.
- **CUDA**: GPU memory is flat 1D, so you typically work with raw pointers: `C[idx] = A[idx] + B[idx]`. If you need multi-dimensional indexing, you must manually compute row/column indices from the flat thread index and handle bounds checking yourself.
- **cute-dsl**: The best of both worlds! Each thread computes a flat `idx`, then uses simple arithmetic (`idx % n` and `idx // n`) to recover 2D coordinates `(mi, ni)`. You then index with clean multi-dimensional syntax: `A[mi, ni]`. Under the hood, CuTe's Layout algebra automatically translates these logical coordinates to the correct physical memory offset — no raw pointer math needed!

## Graduating to Vectorized Execution

GPUs love vectorized operations. Why fetch one float per thread when you can comfortably fetch four? Vectorized pipelines saturate memory bandwidth efficiently.
To vectorize seamlessly, we systematically reshape our datasets using — you guessed it — **tilers**!

```python
@cute.jit 
def vectorized_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Defining a tile of 4 elements wide
    tiler = (1, 4)
    # Zipping it chunks our layout!
    A = cute.zipped_divide(A, tiler)
    B = cute.zipped_divide(B, tiler)
    ...
```

By applying `zipped_divide`, our flat element dimensions transform into chunks of `(1, 4)`. 
The `A`, `B`, and `C` layouts effectively represent shape `((1, 4), (M, N/4))`:
- **Mode 0**: `(1, 4)` is our Inner Tile footprint (yielding 4 elements horizontally).
- **Mode 1**: `(M, N/4)` is our Outer Grid footprint. Each coordinate in the grid points to a vector chunk!

Let's witness the magic in the vectorized kernel:

```python
@cute.kernel
def vectorized_add_kernel(A, B, C):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx

    # Retrieve boundaries from Mode 1 (the grid)
    m, n = A.shape[1]
    ni = idx % n
    mi = idx // n

    # MAGIC HAPPENS HERE
    a_val = A[(None, (mi, ni))].load()
    b_val = B[(None, (mi, ni))].load()

    C[(None, (mi, ni))] = a_val + b_val
```

### Breaking down `A[(None, (mi, ni))]`
This is why we advocated for Zipped Divide earlier! After `zipped_divide`, `A` expects coordinates in the format `[TileCoord, GridCoord]`:
- **`None` for TileCoord**: In CuTe DSL, `None` works like Python's `:` slice — it means "select everything along this mode." Since our tile shape is `(1, 4)`, passing `None` selects all 4 elements of the tile. This gives us back a small tensor (a 4-element vector) rather than a single scalar.
- **`(mi, ni)` for GridCoord**: This pinpoints *which* tile in the `(M, N/4)` grid this thread should access.
- **`.load()`**: This tells the compiler to emit a hardware-level **vectorized memory load** (e.g., `ld.global.v4`), fetching all 4 elements in a single memory transaction directly into registers — far more efficient than 4 separate scalar loads!

### Conclusion
By leveraging Layouts, divide operations, and tilers, `cute-dsl` abstracts away the complex C++ template machinery of raw CUTLASS while preserving its performance. You get optimized layout mapping, zero raw pointer math, and vectorized register loading — all with syntax that feels like writing everyday PyTorch or NumPy code.

Time to fire up your GPUs!
