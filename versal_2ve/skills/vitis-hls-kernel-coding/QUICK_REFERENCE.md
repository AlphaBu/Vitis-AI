# Vitis HLS Kernel — Quick Reference

One-page cheat sheet. Details in [SKILL.md](SKILL.md), [INTERFACES.md](INTERFACES.md), [OPTIMIZATION.md](OPTIMIZATION.md).

## Kernel skeleton

```cpp
#include <cstdint>
#include "ap_int.h"        // ap_int/ap_uint
#include "hls_stream.h"    // hls::stream

extern "C" {                                    // (1) no name mangling
void my_kernel(const int *in, int *out, int n) {
#pragma HLS INTERFACE m_axi     port=in  bundle=gmem0 offset=slave   // (2) data mover
#pragma HLS INTERFACE m_axi     port=out bundle=gmem1 offset=slave
#pragma HLS INTERFACE s_axilite port=in     bundle=control           // (3) pointer offsets
#pragma HLS INTERFACE s_axilite port=out    bundle=control           //     -> same bundle!
#pragma HLS INTERFACE s_axilite port=n      bundle=control           // (4) scalar
#pragma HLS INTERFACE s_axilite port=return bundle=control           // (5) control
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        out[i] = in[i] + 1;
    }
}
}
```

## Interface pragma cheat sheet

| Pragma | Use |
|---|---|
| `INTERFACE m_axi port=P bundle=B offset=slave` | Global-memory pointer (DDR/HBM) |
| `INTERFACE s_axilite port=P bundle=control` | Scalar arg |
| `INTERFACE s_axilite port=return bundle=control` | `ap_ctrl_hs` control (default) |
| `INTERFACE ap_ctrl_chain port=return` | Back-to-back / chained invocations |
| `INTERFACE ap_ctrl_none port=return` | Free-running (streams only) |
| `INTERFACE axis port=P` | AXI4-Stream (kernel-to-kernel) |

`m_axi` tuning: `max_read_burst_length=256 num_read_outstanding=16 latency=64 depth=<cosim>`

## Optimization pragma cheat sheet

| Pragma | Effect |
|---|---|
| `PIPELINE II=1` | Overlap loop iterations (1 result/cycle) |
| `UNROLL [factor=N]` | Parallel copies of loop body (area cost) |
| `DATAFLOW` | Run sub-function calls concurrently (task pipelining) |
| `STREAM variable=s depth=D` | FIFO channel for dataflow |
| `ARRAY_PARTITION variable=A type=complete\|cyclic\|block factor=N dim=d` | More memory ports |
| `ARRAY_RESHAPE ...` | Widen word instead of adding banks |
| `BIND_STORAGE variable=A type=ram_2p impl=uram` | Choose memory kind/ports |
| `AGGREGATE variable=s` | Pack struct into one wide word |
| `PERFORMANCE target_ti=N` | Target interval; tool picks transforms |
| `LATENCY min=a max=b` | Bound latency |
| `LOOP_FLATTEN` | Merge perfect nested loops |
| `LOOP_TRIPCOUNT min= max=` | Reporting-only estimate (variable bounds) |
| `INLINE [off]` | Flatten / keep sub-function |
| `DEPENDENCE variable=A type=inter\|intra direction=RAW dependent=false` | Tell tool a dependency is false |

## Synthesizable? — do / don't

| ✅ Do | ❌ Don't |
|---|---|
| Static arrays, fixed sizes | `malloc`/`new`/`free`/`delete` |
| `ap_int`/`ap_uint`/`ap_fixed` sized to data | Recursion (general) |
| `hls::stream`, `hls_math.h` | System calls, file I/O, threads |
| Pass all I/O via args | Function pointers / virtual functions |
| Compile-time loop bounds (+`LOOP_TRIPCOUNT` if variable) | Global vars as top-level I/O |
| `printf` only in testbench | Rely on `printf` for logic (ignored in synth) |
| One top function, thin wrapper | Bulk arrays on `s_axilite` |

## Decision helper

- **DRAM in/out?** → `m_axi` + `s_axilite` control (`ap_ctrl_hs`).
- **Kernel-to-kernel data?** → `axis` streams; consider `ap_ctrl_none` free-running.
- **II not 1?** → partition arrays / break carried dependency / add operator instances.
- **Low bandwidth?** → separate bundles, sequential access for bursts, widen data path.
- **Overlap memory + compute?** → `DATAFLOW` load→compute→store.

## Build — `make compile` (templates in [BUILD.md](BUILD.md))

```bash
make compile                       # -> build/<KERNEL>.xo (default xilinx_vck190_base_202520_1)
make compile TARGET=hw_emu         # emulation build
make compile PLATFORM=<name|.xpfm> # override platform
```
`Makefile` vars (`?=` overridable): `KERNEL`, `SRC`, `PLATFORM`, `TARGET`, `CFG`, `BUILD_DIR`.
`hls_config.cfg` (via `v++ -c --config`): `[hls]` `clock=3.33ns`, `clock_uncertainty=12%`, `pre_tcl=...`.

Under the hood:
```bash
v++ -c -t hw --platform xilinx_vck190_base_202520_1 --config hls_config.cfg \
    -k my_kernel -o build/my_kernel.xo my_kernel.cpp
v++ -l -t hw --platform xilinx_vck190_base_202520_1 --config link.cfg \
    -o build/app.xclbin build/my_kernel.xo
```
`link.cfg`: `[connectivity] sp=my_kernel_1.in:HBM[0]` / `stream_connect=a.out:b.in`

Standalone HLS: C-sim → C-synth → **co-sim** → export IP.
