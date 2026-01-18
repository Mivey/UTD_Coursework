# Hardware Acceleration of Llama 2 Inference (Vitis HLS)

**Thesis Project:** Design Space Exploration & Memory Hierarchy Optimization for Edge AI

**Thesis Advisor:** Dr. Benjamin Schaefer

## Project Overview
This directory contains the High-Level Synthesis (HLS) source code for a custom hardware accelerator designed to run Llama 2 inference on AMD Versal AI Edge (TE0950) and Zynq UltraScale+ (KR260) platforms.

The core architecture is derived from Andrej Karpathy's `llama2.c` but has been significantly re-architected to exploit the spatial parallelism of FPGA fabrics. The design focuses on overcoming the "Memory Wall" inherent in LLM inference through custom memory hierarchies and burst-transfer optimizations.

## Key Architectural Optimizations
* **Head-Major KV Cache:** Refactored the Key-Value (KV) cache memory layout from Token-Major to **Head-Major**. This enables long burst transfers over the AXI4-Master bus, maximizing DDR bandwidth efficiency during Multi-Head Attention (MHA) calculations.
* **Spatial Parallelism:** Implemented `wide_mha.cpp` to unroll attention mechanisms across parallel compute units, leveraging DSP slices for simultaneous vector operations.
* **Dataflow Pipelining:** Utilized `#pragma HLS DATAFLOW` to overlap compute phases (MatMul, Softmax, RoPE) with memory access operations, minimizing pipeline stalls.
* **Quantization Support:** Integrated `quantizer.h` to handle INT8 weight quantization, reducing memory footprint and enabling deployment on resource-constrained edge devices.

## Source Code Structure

### Core Kernels
| File | Description |
| :--- | :--- |
| **`new_top.cpp`** | Top-level HLS wrapper managing the AXI4-Lite control interface and AXI4-Master data paths. |
| **`mha.cpp`** / **`wide_mha.cpp`** | Multi-Head Attention implementation. `wide_mha.cpp` features aggressive unrolling for high-throughput interfaces (512-bit). |
| **`matmul.cpp`** | Matrix Multiplication kernel optimized for DSP58 (Versal) and DSP48 (Zynq) slices. |
| **`rope.h`** | Rotary Positional Embeddings implementation using CORDIC-like approximations or LUTs. |
| **`rmsnorm.h`** | Root Mean Square Normalization optimized for low-latency execution. |

### Testbenches & Verification
The `testbenches/` directory contains component-level verification setups to ensure functional correctness against the Golden C reference.

* `mha_testbench.cpp`: Validates the attention mechanism logic and memory access patterns.
* `matmul_tb.cpp`: Verifies matrix multiplication accuracy across different quantization scales.
* `quantizer_tb.cpp`: Checks the accuracy of dynamic quantization/dequantization logic.

## Build Instructions
This project is designed for **AMD Vitis HLS 2025.x**.
1. Create a new Vitis HLS project.
2. Add core `.cpp` files as Source.
3. Add `testbenches/*.cpp` files as Test Bench.
4. Set top-level function to `forward` (or `new_top`).
5. Target Part: `xcve2302-sfva784-1LP-e-S` (Versal) or `xck26-sfvc784-2LV-c` (K26/KR260).