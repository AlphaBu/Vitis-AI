---
name: vitis-hls-kernel-coding
description: "Expert assistant for writing PL (Programmable Logic) kernels in C/C++ that compile with Vitis HLS and conform to the Vitis application acceleration framework (XRT / .xo / .xclbin). Covers kernel interface specification (m_axi, s_axilite, axis, block-level control), synthesizable coding restrictions, and optimization (loop pipelining, dataflow, performance pragmas, memory interfaces/bursts, array partitioning). Use when creating, reviewing, or optimizing Vitis HLS PL kernels."
argument-hint: "[kernel_description] [--interface|--restrictions|--optimize|--dataflow|--examples|--help]"
---

# Vitis HLS PL Kernel Coding Skill

Expert guidance for writing high-quality PL kernels in C/C++ that synthesize with **Vitis HLS** (`v++ -c` / `vitis_hls`) and plug into the **Vitis application acceleration flow** (XRT host + `.xo` → `.xclbin`).

> This skill targets the **Vitis kernel flow** (a.k.a. XO/XRT flow), not the Vivado IP flow. The interface rules below are what make a kernel a valid `.xo` that XRT can call.

## Parse Arguments

Extract from `$ARGUMENTS`:
- **Kernel description**: what the kernel should compute (e.g., "matrix multiply", "fir filter", "vector add", "aes stream cipher").
- `--interface`: focus on interface specification (m_axi / s_axilite / axis / control protocol).
- `--restrictions`: focus on synthesizable C/C++ coding constraints.
- `--optimize`: focus on performance optimization (pipeline, unroll, array partition, burst).
- `--dataflow`: focus on task-level parallelism / streaming producer-consumer design.
- `--build`: generate the `Makefile` + `hls_config.cfg` so the user can run `make compile` to produce a `.xo`.
- `--examples`: provide working kernel templates.
- `--help`: show comprehensive guidance.

If no flag is given, infer intent from the description and produce a complete, synthesizable kernel with correct interfaces and sensible optimizations. When you deliver a kernel, also offer/generate the build files (see **[BUILD.md](BUILD.md)**) so it can be compiled with `make compile`.

## Reference Documents

Authoritative sources (consult for exact, version-specific syntax):
- **UG1399** – Vitis HLS User Guide (pragmas, interfaces, coding style, C libraries)
- **UG1393** – Vitis Application Acceleration Development (build flow, linking, memory topology)
- **UG1701** – Embedded Design Using Vitis
- Vitis HLS Tutorials: https://docs.amd.com/r/en-US/Vitis-Tutorials-Vitis-HLS
- Vitis Tutorials (GitHub): https://github.com/Xilinx/Vitis-Tutorials
- HLS Introductory Examples: https://github.com/Xilinx/Vitis-HLS-Introductory-Examples
- Vitis Accel Examples: https://github.com/Xilinx/Vitis_Accel_Examples

## Detailed References (this skill)

- **[INTERFACES.md](INTERFACES.md)** – Full interface specification: m_axi, s_axilite, axis, block-level control protocols, bundling, memory banks, host↔kernel contract.
- **[OPTIMIZATION.md](OPTIMIZATION.md)** – Loop pipeline, dataflow, unroll, array partition, memory bursts, performance pragmas, with worked examples.
- **[BUILD.md](BUILD.md)** – Ready-to-use `Makefile` + `hls_config.cfg` templates so the user can run `make compile` to build a `.xo` (default platform `xilinx_vck190_base_202520_1`); optional link-to-`.xclbin` and `--mode hls` component flow.
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** – One-page pragma cheat sheet and decision checklists.

## Core Responsibilities

1. **Interface design** – Choose and specify the right interfaces so the kernel is a valid `.xo` XRT can control and feed.
2. **Synthesizable coding** – Keep the kernel within the C/C++ subset HLS can synthesize.
3. **Optimization** – Apply pipelining, dataflow, array partitioning, and burst-friendly memory access to hit throughput/latency targets.
4. **Correctness** – Ensure the kernel matches a C-simulation reference and passes co-simulation.

---

## 1. Interface Specification (接口规范)

A Vitis kernel is an ordinary C/C++ function whose **arguments become hardware interfaces** and whose **execution is controlled by XRT**. Three things make a function a valid Vitis kernel:

1. The top function is wrapped in `extern "C" { ... }` (prevents C++ name mangling so XRT/`xrt::kernel` can find it by name).
2. Every pointer/array argument is mapped to an AXI interface (`m_axi` for DRAM/HBM, `axis` for streams).
3. Scalar arguments and the block-level control are mapped to the AXI4-Lite `control` bundle.

### 1.1 Canonical memory-mapped kernel

```cpp
#include <cstdint>

extern "C" {
void vadd(const int *in1,   // input  vector in global memory
          const int *in2,   // input  vector in global memory
          int       *out,   // output vector in global memory
          int        size)  // scalar: number of elements
{
    // ---- Interface pragmas ----
    // Data movers to global memory (DDR/HBM). bundle = physical AXI port.
#pragma HLS INTERFACE m_axi     port=in1  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi     port=in2  bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi     port=out  bundle=gmem0 offset=slave
    // Scalar + pointer base-addresses + control live on AXI4-Lite.
    // Vitis kernel mode requires ALL s_axilite ports in ONE bundle, so map
    // each pointer's offset register (in1/in2/out) to 'control' too:
#pragma HLS INTERFACE s_axilite port=in1             bundle=control
#pragma HLS INTERFACE s_axilite port=in2             bundle=control
#pragma HLS INTERFACE s_axilite port=out             bundle=control
#pragma HLS INTERFACE s_axilite port=size            bundle=control
#pragma HLS INTERFACE s_axilite port=return          bundle=control

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = in1[i] + in2[i];
    }
}
}
```

Notes:
- `offset=slave` puts each pointer's **base address register** in the AXI4-Lite map so the host can set it via `xrt::run` args. This is the standard for the Vitis flow.
- **Single control bundle rule (verified Vitis 2025.2):** the kernel flow requires **all** `s_axilite` ports to share **one** bundle. If you only bundle `size`/`return` as `control` and leave the pointer offsets to inference, they land in a separate bundle (`control_r`) and compilation fails with `[v++ 214-219] ... must be bundled into one bundle`. Fix by explicitly mapping every pointer's offset (`in1`,`in2`,`out`) to the same `control` bundle (as above), or omit all `bundle=` names and let the tool put everything in the default bundle.
- In **recent Vitis versions the tool auto-infers** `s_axilite`/`m_axi` if you omit the pragmas. Writing them explicitly is clearer, but if you name the `control` bundle you must name it on **every** `s_axilite` port (see rule above).
- `port=return` on `s_axilite` enables the software (`ap_ctrl_hs`) control protocol — required for XRT to `start()` the kernel and detect `done`.

### 1.2 Interface types at a glance

| Argument kind | Interface pragma | Purpose |
|---|---|---|
| Pointer / array to global memory | `m_axi` | Burst DDR/HBM access (AXI4 master) |
| Scalar (int/float/enum) | `s_axilite` | Passed once per invocation via control reg |
| `hls::stream<T>` port | `axis` | Kernel-to-kernel / free-running streaming (AXI4-Stream) |
| Block-level control (`return`) | `s_axilite` / `ap_ctrl_*` | Start/done/idle handshake with XRT |

### 1.3 Block-level control protocols

Set on `port=return`:

| Protocol | Pragma | When to use |
|---|---|---|
| `ap_ctrl_hs` (default) | `s_axilite port=return bundle=control` | Standard XRT-controlled kernel (host calls `run.start()`) |
| `ap_ctrl_chain` | `#pragma HLS INTERFACE ap_ctrl_chain port=return` | Kernel called repeatedly / pipelined back-to-back invocations, top-level dataflow |
| `ap_ctrl_none` | `#pragma HLS INTERFACE ap_ctrl_none port=return` | **Free-running kernel** — no software start; driven purely by `axis` streams |

A free-running (`ap_ctrl_none`) kernel must have **only** `axis` (and optionally `ap_none` scalar) interfaces — no `m_axi`, no `s_axilite` control. It is connected via `stream_connect` in the `v++` config.

### 1.4 Streaming kernel example

```cpp
#include "hls_stream.h"
#include "ap_axi_sdata.h"     // for hls::axis / ap_axiu if using side-channels

extern "C" {
void stream_add(hls::stream<int> &in1,
                hls::stream<int> &in2,
                hls::stream<int> &out,
                int size) {
#pragma HLS INTERFACE axis      port=in1
#pragma HLS INTERFACE axis      port=in2
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out.write(in1.read() + in2.read());
    }
}
}
```

See **[INTERFACES.md](INTERFACES.md)** for bundling strategy, memory bank/SLR mapping, burst tuning parameters (`max_read_burst_length`, `num_read_outstanding`, `latency`), and the host-side XRT contract.

---

## 2. Internal Coding Restrictions (内部 coding 限制)

Vitis HLS synthesizes a **subset** of C/C++. Code outside the subset either fails synthesis or is silently ignored. Follow these rules:

### 2.1 Not synthesizable — avoid entirely

- **Dynamic memory**: no `malloc`/`free`/`calloc`/`new`/`delete`. All storage must be statically sized.
- **Recursion**: general recursion is unsupported (bounded template recursion via `struct` metaprogramming is OK).
- **System / OS calls**: no file I/O, no `time()`, no threads, no `getenv`.
- **Function pointers / virtual functions**: not supported (resolve at compile time).
- **Runtime-unbounded constructs**: no unbounded `while(1)` in the compute path unless the kernel is intentionally free-running.

### 2.2 Allowed but constrained

- **`printf`/`std::cout`**: allowed in C-sim/testbench; **ignored during synthesis**. Do not use for logic.
- **Pointers**: must be analyzable. Avoid pointer arithmetic that hides access patterns; avoid pointer-to-pointer and casting between incompatible types. A top-level pointer maps to **one** interface.
- **Standard library**: most `<algorithm>`/`<cmath>` host functions are not synthesizable. Use the **HLS libraries** instead.
- **Loop bounds**: prefer compile-time-constant bounds. For variable bounds, add `#pragma HLS LOOP_TRIPCOUNT min=.. max=..` so the tool can estimate latency (this pragma affects *reporting only*, not hardware).
- **Global variables**: cannot be used as top-level kernel I/O — pass data through arguments.

### 2.3 Use HLS-provided types & libraries

| Need | Use | Header |
|---|---|---|
| Arbitrary-precision integers | `ap_int<N>`, `ap_uint<N>` | `ap_int.h` |
| Fixed-point | `ap_fixed<W,I>`, `ap_ufixed<W,I>` | `ap_fixed.h` |
| Streams (FIFO) | `hls::stream<T>` | `hls_stream.h` |
| Stream w/ side channels | `hls::axis<...>` / `ap_axiu` | `ap_axi_sdata.h` |
| Math | `hls::sqrt`, `hls::exp`, ... | `hls_math.h` |
| Half precision | `half` | `hls_half.h` |
| Vectors | `hls::vector<T,N>` | `hls_vector.h` |

Prefer arbitrary-precision types sized to the data — a 12-bit value should be `ap_uint<12>`, not `int`, to save FF/LUT and improve timing.

### 2.4 Structure the kernel for synthesis

- **One top function** = the kernel; keep it thin (interfaces + orchestration).
- Put compute in **sub-functions**; each becomes an RTL module you can pipeline/inline independently.
- Use the **load → compute → store** pattern (see dataflow below) to decouple memory latency from compute.

See **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** for the full do/don't list.

---

## 3. Optimization Methods (优化方法)

Order of attack: (1) get correct C-sim, (2) pipeline inner loops to II=1, (3) partition arrays feeding pipelines, (4) apply dataflow across load/compute/store, (5) tune bursts/bundles, (6) meet II with `PERFORMANCE`/unroll.

### 3.1 Loop Pipeline

`#pragma HLS PIPELINE` overlaps loop iterations so a new iteration starts every **II** (Initiation Interval) cycles. II=1 = one result/cycle = the goal.

```cpp
for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
    out[i] = f(in[i]);
}
```

- Place on the **innermost** loop you want to run at throughput; HLS auto-unrolls loops nested *inside* a pipelined loop.
- **Pipelining the whole function**: put `#pragma HLS PIPELINE` at the top of the function body → HLS pipelines the top and unrolls all loops.
- **II > 1 causes** (check the schedule/synthesis report): array read/write port limits (fix with `ARRAY_PARTITION`), carried dependencies (fix with `DEPENDENCE`/restructure), resource contention, or long operations (add `BIND_OP`/latency).
- Loops that are not pipelined can be **unrolled** with `#pragma HLS UNROLL [factor=N]` for spatial parallelism (costs area).

### 3.2 Dataflow

`#pragma HLS DATAFLOW` runs sequential sub-function calls as **concurrent tasks** connected by FIFOs/ping-pong buffers — task-level pipelining. This is how you overlap memory read, compute, and write.

```cpp
extern "C" {
void top(const int *in, int *out, int size) {
#pragma HLS INTERFACE m_axi     port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi     port=out bundle=gmem0 offset=slave
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    hls::stream<int> s_in, s_out;
#pragma HLS STREAM variable=s_in  depth=64
#pragma HLS STREAM variable=s_out depth=64

#pragma HLS DATAFLOW
    read_input (in,  s_in,  size);   // task 1
    compute    (s_in, s_out, size);  // task 2
    write_out  (s_out, out, size);   // task 3
}
}
```

Dataflow rules (violations silently serialize the region):
- Each intermediate variable is **produced by exactly one task, consumed by exactly one** (single-producer/single-consumer).
- Use `hls::stream` (or arrays with `#pragma HLS STREAM`) as channels; avoid bypassing a task.
- No conditional execution of tasks inside the dataflow region; keep it a straight-line sequence of calls.
- Size FIFO `depth` to cover producer/consumer rate mismatch and avoid deadlock.

### 3.3 Performance Pragma

`#pragma HLS PERFORMANCE target_ti=<n>` sets a **target initiation interval for a loop/region** and lets HLS choose the transformations (unroll/pipeline) to meet it — a higher-level alternative to hand-tuning. Use it to declare intent ("finish a new transaction every N cycles") and let the tool solve for it; verify the achieved II in the report.

```cpp
loop_rows:
for (int r = 0; r < ROWS; r++) {
#pragma HLS PERFORMANCE target_ti=1
    ...
}
```

Related throughput controls: `LATENCY min/max` (bound loop/function latency), `LOOP_FLATTEN` (merge perfect nested loops), `OCCURRENCE`.

### 3.4 Memory Interfaces (m_axi bursts & bundles)

Global-memory bandwidth usually decides real performance. Optimize the `m_axi` access:

- **Bundle strategy**: each `bundle` = one physical AXI port. Put ports that must be accessed **concurrently** on **different bundles** (parallel bandwidth); group ports on the same bundle when they share a port sequentially (saves resources).
- **Bursting**: access memory **sequentially** so HLS infers bursts. The load→compute→store pattern with a pipelined copy loop produces long bursts automatically. Avoid random/strided access on `m_axi`.
- **Tune the adapter**:
  ```cpp
  #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave \
      max_read_burst_length=256 num_read_outstanding=16 latency=64
  ```
- **Cache/copy to on-chip**: read a tile into a local BRAM/URAM array once, compute from local memory (many fast ports), write results back in a burst.
- Optionally use `hls::burst_maxi` / `hls::stream_of_blocks` for explicit burst control.
- **Memory topology**: which DDR/HBM bank each bundle connects to is chosen at **link time** via `--connectivity.sp kernel_1.in:HBM[0]` in the `v++` config — the kernel just declares bundles.

### 3.5 Array Partition

On-chip arrays default to a BRAM with **2 ports** — a hard limit on how many elements a pipelined loop can read/write per cycle. `#pragma HLS ARRAY_PARTITION` splits an array into multiple memories/registers to create more ports.

```cpp
int buf[16][16];
#pragma HLS ARRAY_PARTITION variable=buf dim=2 type=complete   // 16 elements/row in parallel
// types: complete (registers), cyclic factor=N, block factor=N
```

- `type=complete`: fully split into registers — max parallelism, highest area; use for small arrays / accumulators.
- `type=cyclic factor=N`: interleave into N banks — good for `buf[i]` accessed with stride and for feeding an N-wide unrolled loop.
- `type=block factor=N`: contiguous chunks into N banks.
- `dim=<n>`: which dimension to partition (`dim=0` = all dimensions).
- **Rule**: if a pipelined loop can't reach II=1 because of array port conflicts, partition the array along the accessed dimension by the parallelism factor.
- Related: `#pragma HLS ARRAY_RESHAPE` (widen words instead of adding banks), `#pragma HLS BIND_STORAGE` (choose BRAM/URAM/LUTRAM and #ports), `#pragma HLS AGGREGATE` (pack a struct into one wide word).

See **[OPTIMIZATION.md](OPTIMIZATION.md)** for full worked examples (tiled matmul, FIR, dataflow streaming) and how to read the synthesis/schedule reports.

---

## 4. Build & Verify — `make compile`

Deliver every kernel with a **`Makefile`** and an **`hls_config.cfg`** so it builds with one command. Full copy-paste templates are in **[BUILD.md](BUILD.md)**.

```bash
make compile          # v++ -c ... -> build/<KERNEL>.xo
```

- **Default platform** is `xilinx_vck190_base_202520_1`; override with `make compile PLATFORM=<name-or-.xpfm>`.
- The `Makefile` owns kernel name / source / **platform (default)** / target as overridable `?=` variables and drives `v++ -c`.
- The `hls_config.cfg` (passed via `v++ -c --config`) holds HLS synthesis options (`clock`, `clock_uncertainty`, optional `pre_tcl` directives). Do **not** duplicate `platform`/`target` there — they come from the Makefile.

Underlying commands (what `make` runs / how to link further):
```bash
# Compile HLS kernel to Xilinx Object (.xo)
v++ -c -t hw --platform xilinx_vck190_base_202520_1 --config hls_config.cfg \
    -k vadd -o build/vadd.xo vadd.cpp
# Link into xclbin (memory/stream connectivity via link.cfg)
v++ -l -t hw --platform xilinx_vck190_base_202520_1 --config link.cfg \
    -o build/vadd.xclbin build/vadd.xo
```

Targets: `TARGET=hw` (full synth), `hw_emu`, `sw_emu` (fast iteration). See BUILD.md for the optional `link` target, a `link.cfg`, the standalone `v++ -c --mode hls` component flow, and troubleshooting.

Standalone HLS runs (`vitis_hls -f run.tcl` or `--mode hls`) go: **C-sim → C-synth → co-sim → export**. Always pass C-sim (functional) before trusting synthesis, and run **co-simulation** to confirm the generated RTL matches the C model before hardware.

---

## Workflow When Invoked

1. **Clarify** data types, sizes, and throughput/latency goal if not given.
2. **Pick interfaces**: memory-mapped (`m_axi`) vs streaming (`axis`); decide bundles.
3. **Write a thin `extern "C"` top** with explicit interface pragmas.
4. **Implement compute** in synthesizable sub-functions using HLS types.
5. **Optimize**: pipeline inner loops → partition arrays → dataflow load/compute/store → tune bursts.
6. **Generate build files**: emit the `Makefile` + `hls_config.cfg` from **[BUILD.md](BUILD.md)** (default platform `xilinx_vck190_base_202520_1`) so the user can `make compile` → `.xo`.
7. **Verify**: provide/point to a C testbench for C-sim and recommend co-sim.
8. **Report** expected II/latency and the resource/bandwidth trade-offs made.

Always state which pragmas you added and *why*, and flag any construct that may not synthesize.
