# EEDG 6370: Design & Analysis of Reconfigurable Systems

**Instructor:** Dr. Benjamin Schaefer (Thesis Advisor)

## Course Objectives
A rigorous study of reconfigurable computing architectures and design methodologies. The coursework emphasized hardware-software co-design on heterogeneous SoCs, utilizing the **Intel/Altera Cyclone V** (DE1-SoC) for weekly architectural exploration.

## Featured Project: Hardware-Accelerated Sobel Filter
**Platform:** Digilent Arty Z7 (AMD/Xilinx Zynq-7000 SoC)

This project explores the trade-offs between manual RTL design and High-Level Synthesis (HLS) for image processing workloads.

### Key Technical Achievements
* **RTL vs. HLS Benchmarking:** Developed the Sobel Edge Detection kernel in both **SystemVerilog** and **C++ (Vitis HLS)** to analyze resource utilization and timing closure differences.
* **AXI4-Stream Architecture:** Designed a unified interface for both kernels using **AXI4-Stream**, allowing for modular IP swapping without system redesign.
* **DMA Integration:** Orchestrated high-speed image data transfer between the Processing System (PS) and the FPGA fabric using the **AXI Direct Memory Access (DMA)** engine, optimizing memory bandwidth utilization.