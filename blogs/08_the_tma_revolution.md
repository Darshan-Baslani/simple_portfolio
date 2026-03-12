# The TMA Revolution (Async Copy)

**Difficulty:** Advanced
**Prerequisites:** [Tutorial 04: The Parallel Copy](04_the_parallel_copy.md)

## 1. The Problem (The "Why")
"The CPU is wasting time calculating addresses for copies. Let the hardware do it."

Until now, we have used `TiledCopy` to coordinate warps and threads to fetch data from global memory into shared memory. The problem? **Every single thread is computing memory addresses.** For each element loaded, threads execute instructions just to resolve `address = coord · stride`. This burns registers and arithmetic logic unit (ALU) cycles that should be spent on matrix multiplication.

With the Hopper and Blackwell architectures, NVIDIA introduced the **Tensor Memory Accelerator (TMA)**. Instead of having 128 threads manually calculating pointers and copying data, a single thread can offload the entire tile copy to dedicated hardware.

## 2. The Mental Model (The Visual)
If `TiledCopy` is a warehouse crew of 128 workers carrying boxes (where each worker calculates their path), `TMA` is an **autonomous forklift with a manifest**. 

```text
TiledCopy (threads do address math):

  128 workers compute addresses:
    address = coord · stride
  Then load/store each element.

  [threads] ──address math──▶ LDG/LDS ──▶ smem

TMA (hardware does address math):

  1 dispatcher submits a manifest (descriptor).
  The TMA engine moves the entire pallet.

  [one thread] ──submit manifest──▶ TMA engine ──▶ smem
```

You create the manifest on the host, hand it to the forklift inside the kernel, and the hardware handles the rest asynchronously.

## 3. The Solution (The Code)
Here is a minimal, complete C++ example of copying a 128×64 tile using TMA. Notice how no threads are computing layouts inside the kernel loop!

```Cpp
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cstdio>

using namespace cute;

constexpr int BLK_M = 128;
constexpr int BLK_K = 64;

// Shared memory workspace
struct SharedStorage {
    alignas(128) cutlass::half_t smem[BLK_M * BLK_K];
    uint64_t tma_barrier;
};

template <class TmaDesc>
__global__ void tma_copy_kernel(TmaDesc tma) {
    __shared__ SharedStorage ss;

    // 1. Describe the shared memory target
    auto sA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                 make_stride(Int<BLK_K>{}, Int<1>{}));
    auto sA = make_tensor(make_smem_ptr(ss.smem), sA_layout);

    // 2. Fetch the global memory tensor from the TMA descriptor
    auto gA = tma.get_tma_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{}));

    // 3. Partition the TMA copy
    auto [tAgA, tAsA] = tma_partition(
        tma,
        Int<0>{},             // CTA block coordinate
        Layout<_1>{},         // Grid layout
        group_modes<0,2>(sA), // Collapse modes for flat copy
        group_modes<0,2>(gA)
    );

    // 4. Dispatch the autonomous forklift!
    if (threadIdx.x == 0) {
        using Bar = cutlass::arch::ClusterTransactionBarrier;
        
        // Initialize the mbarrier
        Bar::init(&ss.tma_barrier, 1);
        int bytes_to_copy = BLK_M * BLK_K * sizeof(cutlass::half_t);
        
        // Announce the expected transaction size
        Bar::arrive_and_expect_tx(&ss.tma_barrier, bytes_to_copy);
        
        // Issue the async copy using the manifest
        copy(tma.with(ss.tma_barrier), tAgA, tAsA);
        
        // Block until the forklift delivers the pallet
        Bar::wait(&ss.tma_barrier, 0);
    }
    __syncthreads();
    
    // Now sA is filled and ready for computation!
}

void host_launch_example() {
    // We must build the TMA descriptor on the HOST.
    cutlass::half_t* d_A;
    cudaMalloc(&d_A, BLK_M * BLK_K * sizeof(cutlass::half_t));

    auto mA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                 make_stride(Int<BLK_K>{}, Int<1>{}));
    auto mA = make_tensor(make_gmem_ptr(d_A), mA_layout);
    
    auto sA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                 make_stride(Int<BLK_K>{}, Int<1>{}));

    // Printing the manifest for the autonomous forklift
    auto tma = make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout,
                             make_shape(Int<BLK_M>{}, Int<BLK_K>{}));

    // Launch the kernel
    tma_copy_kernel<<<1, 128>>>(tma);
    cudaDeviceSynchronize();
    cudaFree(d_A);
}

int main() {
    host_launch_example();
    printf("TMA Copy Successful!\n");
    return 0;
}
```

## 4. Step-by-Step Explanation
Line `make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, ...)`: This runs entirely on the host. It bundles the `Layout<Shape, Stride>` math into a hardware-readable manifest. We are telling the GPU how global memory and shared memory relate, *before* the kernel starts.

Line `auto [tAgA, tAsA] = tma_partition(...)`: We slice up the tensors for the copy. Unlike `TiledCopy` which assigns slices per-thread, `tma_partition` treats the CTA as one large worker.

Line `if (threadIdx.x == 0)`: Only **one dispatcher thread** is needed to initiate the TMA load. 

Line `copy(tma.with(&ss.tma_barrier), tAgA, tAsA)`: The actual dispatch command. The hardware asynchronously begins fetching the entire tile into shared memory.

Line `Bar::wait(...)`: Because the forklift works asynchronously, we must wait at the loading dock (the `mbarrier`) before we can safely read the data.

## 5. Engineer's Notebook (Latent Space Notes)
**Analogy:** `Tensor Memory Accelerator (TMA)` is an **autonomous forklift with a manifest**. You don’t have 128 workers carrying boxes anymore. One dispatcher hands the forklift a manifest (the TMA descriptor), and the hardware moves the entire pallet into shared memory. `make_tma_atom` is printing the manifest for the autonomous forklift (the TMA descriptor).

**Hardware Constraints & Gotchas:**
> **Gotcha — TMA descriptors are host-built.** `make_tma_atom` must run on the CPU. The descriptor encodes the address math for the tile and cannot be created inside the kernel.

> **Gotcha — TMA is async and barrier-driven.** One thread launches the copy; an mbarrier (`ClusterTransactionBarrier`) is required before any thread reads from smem. Without the `wait()`, you will read garbage data.

> **Gotcha — TMA prefers static layouts.** Tile shape and strides should be `Int<N>{}` so the descriptor is fully static and the copy is vector-friendly. CuTe uses these static layouts at compile-time to guarantee contiguity and optimize the transfer.

> **B200/Hopper Note:** TMA is the primary way to saturate memory bandwidth on SM90+ architectures. Understanding strides (`address = coord · stride`) is the only way to program the TMA descriptor correctly!
