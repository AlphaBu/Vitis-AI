# vitis-hls-kernel-coding

Expert assistant skill for writing **PL (Programmable Logic) kernels** in C/C++ that compile with **Vitis HLS** and conform to the **Vitis application acceleration framework** (XRT host → `.xo` → `.xclbin`).

## What it covers

- **接口规范 (Interface specification)** — `m_axi` global memory, `s_axilite` scalars/control, `axis` streams, and block-level control protocols (`ap_ctrl_hs` / `ap_ctrl_chain` / `ap_ctrl_none`), bundling, memory banks, and the host↔kernel XRT contract.
- **内部 coding 限制 (Coding restrictions)** — the synthesizable C/C++ subset, HLS types/libraries (`ap_int`, `ap_fixed`, `hls::stream`, `hls_math`), and kernel structuring rules.
- **优化方法 (Optimization)** — Loop Pipeline, Dataflow, Performance Pragma, Memory Interfaces / bursts, and Array Partition, with worked examples and report-reading guidance.
- **编译 (Build)** — `Makefile` + `hls_config.cfg` templates so `make compile` builds the kernel to a `.xo` (default platform `xilinx_vck190_base_202520_1`).

## Files

| File | Purpose |
|---|---|
| `SKILL.md` | Main entry: interface spec, coding restrictions, optimization overview, workflow |
| `INTERFACES.md` | Detailed interface reference + host XRT contract + pitfalls |
| `OPTIMIZATION.md` | Detailed optimization techniques with worked examples |
| `BUILD.md` | `Makefile` + `hls_config.cfg` templates for `make compile` → `.xo` (+ link, `--mode hls`) |
| `QUICK_REFERENCE.md` | One-page pragma cheat sheet + do/don't + build commands |

## Usage

```
/vitis-hls-kernel-coding <kernel_description> [flags]
```

Flags: `--interface`, `--restrictions`, `--optimize`, `--dataflow`, `--build`, `--examples`, `--help`.

Examples:
- `/vitis-hls-kernel-coding vector add with m_axi interfaces`
- `/vitis-hls-kernel-coding tiled matrix multiply --optimize`
- `/vitis-hls-kernel-coding streaming fir filter --dataflow`
- `/vitis-hls-kernel-coding vector add --build` (also emit `Makefile` + `hls_config.cfg` for `make compile`)

## Reference documents

- UG1399 – Vitis HLS User Guide
- UG1393 – Vitis Application Acceleration Development
- UG1701 – Embedded Design Using Vitis
- [Vitis HLS Tutorials](https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-HLS) ·
  [Vitis Tutorials (GitHub)](https://github.com/Xilinx/Vitis-Tutorials) ·
  [HLS Introductory Examples](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples) ·
  [Vitis Accel Examples](https://github.com/Xilinx/Vitis_Accel_Examples)

## Scope note

Targets the **Vitis kernel (XO/XRT) flow**, not the Vivado IP flow. This skill covers PL kernels only; for AI Engine (AIE/NPU) kernels see the `aie-kernel-coding` skill.
