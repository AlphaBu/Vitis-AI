# Vitis HLS Kernel Interface Specification (Detailed)

Complete reference for making an HLS function a valid Vitis kernel (`.xo`) that XRT can control and feed. Companion to [SKILL.md](SKILL.md).

---

## 1. The kernel contract

A Vitis kernel is a C/C++ function where:

1. **The top function is `extern "C"`** so XRT resolves it by name (`xrt::kernel(device, uuid, "vadd")`).
2. **Each argument maps to a hardware interface** via `#pragma HLS INTERFACE`.
3. **Block-level control** (`port=return`) defines how the host starts/monitors the kernel.

```
Host (XRT)  --AXI4-Lite (control)-->  [kernel control regs + scalar args + pointer base addrs]
Host DRAM   <--AXI4 (m_axi)-------->  [kernel data movers]
Kernel      <--AXI4-Stream (axis)-->  other kernels (streaming)
```

The three argument-to-interface mappings:

| C argument | Interface | Direction/role |
|---|---|---|
| `T *ptr`, `T arr[N]` | `m_axi` (data) + `s_axilite` (base addr) | Bulk global memory access |
| `int n`, `float k` | `s_axilite` | Scalar, set once per run |
| `hls::stream<T> &s` | `axis` | Streaming FIFO to/from another kernel |
| `return` | `s_axilite` / `ap_ctrl_*` | Start/done/idle/ready handshake |

---

## 2. `m_axi` — memory-mapped master (global memory)

Used for pointer/array args that read/write DDR or HBM.

```cpp
#pragma HLS INTERFACE m_axi port=<arg> bundle=<name> offset=slave \
    depth=<n> \
    max_read_burst_length=<2..256> \
    max_write_burst_length=<2..256> \
    num_read_outstanding=<n> \
    num_write_outstanding=<n> \
    latency=<cycles>
```

| Option | Meaning | Guidance |
|---|---|---|
| `bundle` | Groups ports onto one physical AXI port | Separate bundles → concurrent bandwidth; shared bundle → resource saving, sequential sharing |
| `offset=slave` | Put the base-address register in the AXI4-Lite `control` map | **Standard for Vitis flow** — host sets addr via `run` arg |
| `max_read_burst_length` | Max beats per read burst (powers of 2, 2–256) | Match to your longest sequential run; 64–256 for streaming copies |
| `num_read_outstanding` | In-flight read bursts the adapter buffers | Increase to hide DRAM latency (costs BRAM for the buffer) |
| `latency` | Expected memory latency HLS schedules around | Raise (e.g. 64) for HBM/DDR to allow deeper pipelining |
| `depth` | Element count for **co-simulation** only | Set to the max transfer size so co-sim allocates enough |

### Bundling strategy

- Ports on the **same** `bundle` share one AXI master → their transactions are **serialized** through that port.
- Ports on **different** bundles get **independent** AXI masters → true parallel bandwidth.
- Rule of thumb: put each concurrently-accessed input/output on its own bundle, up to the platform's available AXI ports / memory banks.

```cpp
// in1 & in2 read in parallel, out written in parallel → 3 bundles
#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave
```

### Bursting (critical for bandwidth)

HLS infers a burst when accesses are **sequential and monotonic** inside a pipelined loop. To maximize burst inference:

```cpp
// GOOD: sequential, burst-inferable copy into local memory
copy: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
    local[i] = in[i];          // one long burst
}
// BAD: random/strided access → no burst, single-beat transfers
for (int i = 0; i < N; i++) out[perm[i]] = ...;   // scattered
```

- Read a tile into on-chip memory in one burst, compute locally, write back in one burst (see load/compute/store below).
- For explicit control use `hls::burst_maxi<T>` or `memcpy`-style copies.
- Verify actual burst length in the synthesis report ("Burst" / "AXI" section) and via `v++ --profile`.

---

## 3. `s_axilite` — control & scalars

The AXI4-Lite `control` bundle carries: block-level control regs, scalar arguments, and the base-address registers of `offset=slave` pointers.

```cpp
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control   // enables ap_ctrl_hs
```

- **All `s_axilite` ports must share one bundle** (conventionally `control`) in the Vitis flow. This includes the base-address offset register of every `offset=slave` `m_axi` pointer. **Verified (Vitis 2025.2): write the `s_axilite ... bundle=control` pragma explicitly for each pointer** — if you rely on auto-inference the tool may place the offsets in a separate bundle (e.g. `control_r`) and linking fails with `[v++ 214-219] all s_axilite ports must be bundled into one bundle`.
- Scalars are written by the host before `start()` and are constant for the whole invocation.
- Do **not** put a large array on `s_axilite` for bulk data — it is a single-word slave, extremely slow. Use `m_axi`.

### Auto-inference (recent Vitis)

Modern Vitis auto-adds `s_axilite` for scalars/pointer-offsets and `m_axi` for pointers when pragmas are omitted. **Best practice: still write them explicitly** — you must control `bundle` grouping and burst parameters, and explicit pragmas make review/porting reliable.

---

## 4. `axis` — AXI4-Stream (kernel-to-kernel)

For `hls::stream<T>` arguments used to stream data between kernels (no DRAM round-trip).

```cpp
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
```

- Backed by `hls::stream<T>` (data only) or `hls::axis<T,User,Id,Dest>` / `ap_axiu<...>` when you need TLAST/TKEEP/TUSER side channels.
- Stream connectivity between kernels is set at **link time**:
  ```
  # link.cfg
  [connectivity]
  stream_connect=producer_1.out_stream:consumer_1.in_stream
  ```
- Streams give continuous throughput and are the basis for **free-running** kernels.

### `ap_axiu` example with TLAST

```cpp
#include "ap_axi_sdata.h"
typedef ap_axiu<32,0,0,0> pkt;   // 32-bit data + TLAST/TKEEP/TSTRB

extern "C" void filt(hls::stream<pkt> &in, hls::stream<pkt> &out, int n) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=n      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        pkt t = in.read();
        t.data = t.data * 3;
        out.write(t);           // t.last carried through
    }
}
```

---

## 5. Block-level control protocols (`port=return`)

Defines how execution starts and completion is signaled.

| Protocol | How to set | Behavior | Use case |
|---|---|---|---|
| **`ap_ctrl_hs`** (default) | `s_axilite port=return bundle=control` | Host writes AP_START; kernel raises AP_DONE/AP_IDLE/AP_READY | Standard XRT-controlled kernel |
| **`ap_ctrl_chain`** | `#pragma HLS INTERFACE ap_ctrl_chain port=return` | Adds `ap_continue` — kernel can be restarted back-to-back before draining | Repeated invocations, kernel used inside a larger dataflow, higher throughput chaining |
| **`ap_ctrl_none`** | `#pragma HLS INTERFACE ap_ctrl_none port=return` | No control handshake at all | **Free-running** kernel driven only by streams |

### Free-running kernel requirements (`ap_ctrl_none`)

- **No** `m_axi`, **no** `s_axilite` — only `axis` ports (and optionally `ap_none` scalars tied off at link).
- Typically an infinite loop that always reads its input streams:
  ```cpp
  extern "C" void freerun(hls::stream<pkt>&in, hls::stream<pkt>&out){
  #pragma HLS INTERFACE axis port=in
  #pragma HLS INTERFACE axis port=out
  #pragma HLS INTERFACE ap_ctrl_none port=return
      while (true) {
  #pragma HLS PIPELINE II=1
          pkt t = in.read();          // blocks until data
          t.data += 1;
          out.write(t);
      }
  }
  ```
- Instantiated with `nk` and wired with `stream_connect`; XRT does not start/stop it.

---

## 6. Complete annotated template (memory-mapped)

```cpp
#include <cstdint>
#include "ap_int.h"

extern "C" {
void mmult(const ap_int<16> *A,   // MxK
           const ap_int<16> *B,   // KxN
           ap_int<32>       *C,   // MxN
           int M, int N, int K) {
    // --- Data movers ---
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave max_read_burst_length=256 num_read_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 offset=slave max_read_burst_length=256 num_read_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 offset=slave max_write_burst_length=256 num_write_outstanding=16
    // --- Pointer base-address offsets: MUST share the one control bundle ---
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=B      bundle=control
#pragma HLS INTERFACE s_axilite port=C      bundle=control
    // --- Scalars + control ---
#pragma HLS INTERFACE s_axilite port=M      bundle=control
#pragma HLS INTERFACE s_axilite port=N      bundle=control
#pragma HLS INTERFACE s_axilite port=K      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    // ... tiled implementation ...
}
}
```

---

## 7. Host-side (XRT) contract — what the interfaces imply

The interface choices above determine the host call sequence:

```cpp
auto krnl = xrt::kernel(device, uuid, "vadd");
auto bo_in = xrt::bo(device, bytes, krnl.group_id(0)); // group_id from bundle→bank
// ... fill & sync bo ...
auto run = krnl(bo_in1, bo_in2, bo_out, size);   // arg order == C arg order
run.wait();                                       // waits for AP_DONE (ap_ctrl_hs)
```

Key correspondences:
- **C argument order == host `krnl(...)` argument order.** Pointers pass as `xrt::bo`, scalars pass by value.
- `krnl.group_id(i)` for a `bo` must match the memory bank the pointer's **bundle** is linked to (`--connectivity.sp`).
- `run.wait()` relies on `ap_ctrl_hs`/`ap_ctrl_chain`; a free-running kernel has no `run` to wait on.

---

## 8. Interface pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Kernel name not found by XRT | Missing `extern "C"` | Wrap top function in `extern "C" {}` |
| Low DRAM bandwidth | Ports share one `bundle`, or strided access | Split bundles; make access sequential for bursts |
| Host hangs on `run.wait()` | Wrong control protocol / kernel never asserts DONE | Ensure `s_axilite port=return`; no unbounded loop |
| `group_id`/bank mismatch error at runtime | `bo` bank ≠ bundle's linked bank | Align `--connectivity.sp` with `group_id` |
| Huge/slow control transfers | Bulk array on `s_axilite` | Move bulk data to `m_axi` |
| Free-running kernel won't link | Has `m_axi`/`s_axilite` with `ap_ctrl_none` | Use only `axis` interfaces |
