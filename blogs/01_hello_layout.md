# Hello, Layout! — Visualizing Memory in CuTe

**Difficulty:** Beginner
**Prerequisites:** Basic CUDA (kernel launch, `__global__`), C/C++ pointers

## 1. The Problem (The "Why")

You have a flat `float*` pointer — 1D memory. But your algorithm thinks in rows and columns: a matrix, a tile, a 2D block of pixels. Right now you're hand-writing index math like `ptr[row * WIDTH + col]` and praying you got the constant right. Every time the layout changes (transpose? padding? different tile size?), you rewrite the formula and re-introduce bugs.

CuTe's `Layout` solves this by making the coordinate-to-address mapping a **first-class object**. You declare the shape (how many rows, how many columns) and the stride (the distance between elements), and the Layout *computes* the flat index for you. Change from row-major to column-major? Swap the strides. Reshape from 2D to 3D? Add a mode. The algorithm code never changes.

> **B200 Note:** On Hopper and Blackwell GPUs, the TMA (Tensor Memory Accelerator) hardware needs a *descriptor* that encodes shape and stride information. Understanding Layouts is the **only** way to program TMA correctly.

## 2. The Mental Model (The Visual)

A `Layout` is a **lens** that lets you view flat, 1D memory as if it were a multidimensional grid.

Take `Shape<2, 4>` — that's 2 rows and 4 columns, so 8 elements total. The `Stride` decides *how* those coordinates map to physical addresses.

### Column-Major (The Default): `(2,4):(1,2)`

Stride `(1, 2)` means: move 1 row down → address +1. Move 1 column right → address +2.

```text
Physical Memory (1D):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘

Layout maps coordinates → addresses:

        col 0   col 1   col 2   col 3
       ┌───────┬───────┬───────┬───────┐
row 0  │   0   │   2   │   4   │   6   │
       ├───────┼───────┼───────┼───────┤
row 1  │   1   │   3   │   5   │   7   │
       └───────┴───────┴───────┴───────┘

address = row * 1  +  col * 2
          ───────     ───────
          stride[0]   stride[1]
```

Notice the columns are stored contiguously in memory: addresses `0,1` are column 0, addresses `2,3` are column 1, etc. That's **column-major** — just like Fortran and MATLAB.

### Row-Major: `(2,4):(4,1)`

Stride `(4, 1)` means: move 1 row down → address +4. Move 1 column right → address +1.

```text
        col 0   col 1   col 2   col 3
       ┌───────┬───────┬───────┬───────┐
row 0  │   0   │   1   │   2   │   3   │
       ├───────┼───────┼───────┼───────┤
row 1  │   4   │   5   │   6   │   7   │
       └───────┴───────┴───────┴───────┘

address = row * 4  +  col * 1
          ───────     ───────
          stride[0]   stride[1]
```

Now rows are contiguous: addresses `0,1,2,3` are row 0. That's **row-major** — the C/C++ convention.

> **Key Insight:** Same shape, same data, same 8 elements. The *only* difference is the stride. That's the entire power of a Layout: **the shape says "what," the stride says "where."**

## 3. The Solution (The Code)

A complete CUDA kernel that creates Layouts and prints the coordinate-to-address mapping.

```cpp
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// Print a rank-2 layout as a grid of (row, col) -> address
template <class Shape, class Stride>
__device__ void print_layout_2d(Layout<Shape, Stride> const& layout)
{
    // Only let thread 0 print
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("Layout: ");
    print(layout);
    printf("\n");

    for (int row = 0; row < size<0>(layout); ++row) {
        for (int col = 0; col < size<1>(layout); ++col) {
            printf("  (%d,%d)->%d", row, col, int(layout(row, col)));
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void hello_layout_kernel()
{
    // 1. Column-major (default when stride is omitted)
    auto col_major = make_layout(make_shape(Int<2>{}, Int<4>{}));
    printf("=== Column-Major ===\n");
    print_layout_2d(col_major);

    // 2. Row-major (use LayoutRight tag)
    auto row_major = make_layout(make_shape(Int<2>{}, Int<4>{}), LayoutRight{});
    printf("=== Row-Major ===\n");
    print_layout_2d(row_major);

    // 3. Custom stride
    auto custom = make_layout(make_shape(Int<2>{}, Int<4>{}),
                              make_stride(Int<8>{}, Int<1>{}));
    printf("=== Custom Stride (2,4):(8,1) ===\n");
    print_layout_2d(custom);
}

int main()
{
    hello_layout_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**Expected Output:**

```text
=== Column-Major ===
Layout: (_2,_4):(_1,_2)
  (0,0)->0  (0,1)->2  (0,2)->4  (0,3)->6
  (1,0)->1  (1,1)->3  (1,2)->5  (1,3)->7

=== Row-Major ===
Layout: (_2,_4):(_4,_1)
  (0,0)->0  (0,1)->1  (0,2)->2  (0,3)->3
  (1,0)->4  (1,1)->5  (1,2)->6  (1,3)->7

=== Custom Stride (2,4):(8,1) ===
Layout: (_2,_4):(_8,_1)
  (0,0)->0  (0,1)->1  (0,2)->2  (0,3)->3
  (1,0)->8  (1,1)->9  (1,2)->10  (1,3)->11
```

## 4. Step-by-Step Explanation

**Line: `auto col_major = make_layout(make_shape(Int<2>{}, Int<4>{}));`**

- `make_shape(Int<2>{}, Int<4>{})` creates a compile-time shape of 2 rows by 4 columns. `Int<N>{}` is CuTe's static integer — the compiler knows the value, so the stride math becomes zero-cost.
- `make_layout(shape)` with no stride argument defaults to `LayoutLeft` (column-major). CuTe computes the stride as an exclusive prefix product of the shape from left to right: stride₀ = 1, stride₁ = 1 × 2 = 2. Result: `(2,4):(1,2)`.

**Line: `auto row_major = make_layout(make_shape(Int<2>{}, Int<4>{}), LayoutRight{});`**

- The `LayoutRight{}` tag generates strides as an exclusive prefix product from *right to left*: stride₁ = 1, stride₀ = 1 × 4 = 4. Result: `(2,4):(4,1)`.

**Line: `auto custom = make_layout(make_shape(...), make_stride(Int<8>{}, Int<1>{}));`**

- Here we provide an explicit stride. Row stride of 8 means there's a gap between rows — this is how you express a matrix that lives inside a larger allocation (e.g., a 2×4 submatrix of an 8-column matrix).

**Line: `layout(row, col)`**

- This is the layout's core operation: it takes a 2D coordinate and returns the flat 1D index. The formula is `row × stride[0] + col × stride[1]`. For column-major: `1×row + 2×col`. For row-major: `4×row + 1×col`.

**Line: `size<0>(layout)` / `size<1>(layout)`**

- `size<0>` returns the extent of mode 0 (rows = 2). `size<1>` returns mode 1 (columns = 4). This drives the loop bounds. Because the shape is static, the compiler can fully unroll these loops.

## 5. Engineer's Notebook (Latent Space Notes)

**Analogy:** A Layout is a *spreadsheet formula* for memory. The Shape is how many rows and columns the spreadsheet has. The Stride is the formula in each cell that says "to find my data, take my row number × *this* + my column number × *that*." Change the formula, change the order data is fetched from RAM — but the spreadsheet looks identical to the user.

**Hardware Note:** On B200/Hopper GPUs, when you set up a TMA descriptor with `make_tma_copy`, you must provide a Layout that describes the tensor's shape and strides in global memory. If you get the strides wrong, the TMA engine will fetch garbage. The concepts from this tutorial — `make_shape`, `make_stride`, column-major vs. row-major — are *exactly* what the TMA descriptor encodes.

**CuTe Notation Cheat-Sheet:**

| Notation | Meaning |
| :--- | :--- |
| `(2,4):(1,2)` | Shape `(2,4)`, Stride `(1,2)` — column-major |
| `(2,4):(4,1)` | Shape `(2,4)`, Stride `(4,1)` — row-major |
| `_N` in output | Static (compile-time) integer |
| `N` in output | Dynamic (run-time) integer |
| `Int<N>{}` | C++ way to write a static integer |
| `LayoutLeft` | Column-major stride generation (default) |
| `LayoutRight` | Row-major stride generation |
