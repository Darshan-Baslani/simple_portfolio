# 09. WGMMA (Warpgroup MMA)

**Difficulty:** Advanced
**Prerequisites:** Tutorial 06: Hello, MMA, Tutorial 08: The TMA Revolution

## 1. The Problem (The "Why")
Single warps are too small. We need 128 threads driving the Tensor Core together.
In previous generations, threads had to load memory from shared memory into their own registers before they could perform an MMA (Matrix Multiply Accumulate). This creates a massive bottleneck: the register file gets saturated, and the `ldmatrix` instructions waste precious clock cycles. We need a way to feed the Tensor Core directly from the shared memory staging area where the TMA forklift just dropped our data.

## 2. The Mental Model (The Visual)
WGMMA is the **Heavy Industrial Press**. Unlike a standard press where each worker loads their own tray, the WGMMA press reads directly from the forklift's delivery zone.

```text
Single-Warp MMA (SM80/SM89):
  smem ──(ldmatrix)──▶ registers ──(mma.sync)──▶ accumulators

WGMMA (SM90):
  smem ──(wgmma descriptor)────────────────────▶ accumulators
           (128 threads coordinate)
```

And to make things even more efficient, we have **TMA Multicast** — the forklift driver shouting "who else needs this?" and dropping one pallet into multiple CTA factories at once:

```text
  Global Memory
       │ (TMA Forklift reads once)
       ▼
  Cluster Broadcast Network
    ┌──┴──┐
    ▼     ▼
  smem  smem    (Delivered to multiple CTA factories simultaneously)
  CTA0  CTA1
```

## 3. The Solution (The Code)
Here is how you define a WGMMA instruction using CuTe:

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

void example() {
    // 1. Define the Heavy Industrial Press (WGMMA)
    // 64x64x16 atom. F32 accumulators, F16 inputs. 
    // SS = both A and B are read directly from Shared Memory.
    // GMMA::Major::K means the K dimension is contiguous (TN layout).
    using WGMMA_Atom = SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
    
    // 2. Build the factory floor plan for the 4-warp crew
    auto tiled_mma = make_tiled_mma(WGMMA_Atom{});
    
    // 3. Assume smem_A and smem_B are descriptors to our shared memory
    // (In reality, you would build these using make_smdest_ptr)
    // auto frag_A = ... 
    // auto frag_B = ...
    // auto accum  = ...
    
    // 4. Press the button! (Requires exactly 128 threads)
    // gemm(tiled_mma, frag_A, frag_B, accum);
}
```

## 4. Step-by-Step Explanation
Line 14: `SM90_64x64x16_F32F16F16_SS<...>` defines our specific `WGMMA` operation. It calculates a 64x64x16 matrix multiply. Crucially, the `_SS` suffix means both A and B operands will be sourced directly from Shared Memory descriptors, not from registers. The template parameters `<GMMA::Major::K, GMMA::Major::K>` indicate that both A and B have K as their fast-moving contiguous dimension.

Line 17: `make_tiled_mma` binds the instruction to a `Warpgroup`. CuTe knows that this SM90 atom requires exactly 128 threads (4 warps) and configures the `TiledMMA` layout automatically to coordinate them.

Line 26: `gemm(...)` dispatches the asynchronous `wgmma` instruction to the hardware. All 128 threads must participate in this call.

## 5. Engineer's Notebook (Latent Space Notes)
Analogy: Think of the `Warpgroup` as a synchronized crew of 4 warps working together as a single unit, and `SM90_WGMMA` as the Heavy Industrial Press. It takes inputs directly from the forklift's delivery zone (`smem` desc) rather than personal trays. `tma_multicast` is the forklift driver shouting "who else needs this?", delivering one pallet to multiple factories simultaneously.

Hardware Note: On Hopper (SM90), WGMMA requires exactly 128 threads. You cannot launch a warpgroup MMA with a single warp; it is a collaborative operation across 4 warps. TMA Multicast requires Threadblock Clusters — you can only multicast a TMA load to CTAs that are part of the same hardware cluster.