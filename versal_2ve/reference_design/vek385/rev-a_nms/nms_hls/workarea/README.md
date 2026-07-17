# ONNX NonMaxSuppression Vitis HLS Design

This directory contains a Vitis HLS implementation of the ONNX
`NonMaxSuppression` operator. The current experiment targets a Versal AI Core
device and is configured for one batch, two score classes, and up to 6,300
boxes.

## Current build configuration

The source has fallback capacities:

```cpp
MAX_BOXES   = 4000
NUM_CLASSES = 80
NUM_BATCHES = 1
```

The experiment does not use the first two fallback values. `run.tcl` overrides
them for both the kernel and testbench:

```tcl
set defs {-DMAX_BOXES=6300 -DNUM_CLASSES=2}
```

The synthesized configuration is therefore:

| Parameter | Value |
|---|---:|
| `MAX_BOXES` | 6,300 |
| `NUM_CLASSES` | 2 |
| `NUM_BATCHES` | 1 |
| Target part | `xcvc1902-vsva2197-2MP-e-S` |
| Target clock | 3.3 ns |
| Vitis HLS | 2026.1 |

## Files

| Path | Purpose |
|---|---|
| `rearchitect/v1/src/kernel_hls.cpp` | HLS kernel implementation |
| `rearchitect/v1/src/kernel_hls.h` | Kernel declaration and BF16 type |
| `rearchitect/v1/tb/testbench.cpp` | C simulation and co-simulation testbench |
| `run.tcl` | Vitis HLS build script |
| `nms_onnx_component/hls/syn/report/` | C-synthesis reports |
| `nms_onnx_component/hls/sim/report/` | RTL co-simulation reports |
| `reports/` | Copied summary and implementation reports from earlier runs |

## Kernel interface

The top function is:

```cpp
void nms_onnx(
    const bf16_t *boxes_in,
    const bf16_t *scores_in,
    int32_t      *selected_out,
    int           num_batches,
    int           num_classes,
    int           num_boxes,
    int           max_out_per_class,
    bf16_t        iou_threshold,
    bf16_t        score_threshold,
    int           center_point_box,
    int32_t      *num_selected);
```

`bf16_t` is:

```cpp
using bf16_t = ap_float<16, 8>;
```

Boxes, scores, and the two thresholds enter the kernel as BF16. Boxes and
scores are expanded to FP32 during input loading; thresholds are expanded
before `compute` starts. All current on-chip box/score storage, sorting keys,
and IoU arithmetic remain FP32.

The output is an array of integer triplets:

```text
[batch_index, class_index, box_index]
```

`num_selected[0]` gives the number of valid triplets.

## Processing architecture

For every batch and class, the kernel performs:

1. Filter scores using `score_threshold`.
2. Sort candidate indices by descending score with a stable four-pass,
   8-bit-digit LSD radix sort.
3. Gather box coordinates into sorted-rank order.
4. Run greedy NMS.
5. Write selected triplets to the output buffer.

The greedy loop is sequential at the selected-box level because each accepted
box updates suppression state before the next accepted box can be determined.
The suppression scan inside each greedy iteration is parallelized across eight
IoU lanes.

Important implementation details:

- Suppression state is indexed by sorted candidate rank rather than original
  box index.
- Sorted coordinate arrays use cyclic partitioning with factor 8.
- The IoU scan is unrolled by factor 8.
- C synthesis reports eight independent IoU engines and an achieved IoU-loop
  initiation interval of 1.
- Suppression-state initialization is fused into the sorted-coordinate gather.
- Top-level `DATAFLOW` separates load, compute, and store stages.

The divide in the IoU test is removed:

```text
intersection / union > threshold
```

is evaluated as:

```text
intersection > threshold * union
```

when the union is positive.

## Testbench workloads

Four tests are active in `main()`.

### Corner-format test

- One batch
- One class
- Six boxes
- Corner representation: `[y1, x1, y2, x2]`
- Checks score ordering, suppression, and output triplet order

### Center-format test

- One batch
- One class
- Four boxes
- Center representation: `[cx, cy, width, height]`
- Checks center-to-corner conversion and score-threshold filtering

### Case A: hard score filtering

Case A uses:

- One batch
- Two classes
- 6,300 boxes
- Non-overlapping boxes
- Only 100 boxes per class pass `score_threshold=0.25`
- Approximately 200 total selected boxes

All input boxes and scores are transferred, but only about 100 candidates per
class enter sorting and greedy NMS. This case emphasizes input loading and
filtering while keeping sort and greedy work relatively small.

### Case B: heavy geometric suppression

Case B uses:

- One batch
- Two classes
- 6,300 boxes
- All 6,300 boxes per class pass the score threshold
- 100 spatial clusters
- 63 identical boxes per cluster
- Boxes from different clusters do not overlap
- Boxes within a cluster have IoU 1.0
- One box survives from each cluster
- Approximately 100 selected boxes per class, 200 total

Case B keeps filtering and sorting at full size. For every accepted cluster
representative, the inner greedy scan traverses the remaining sorted candidate
ranks. Many already-suppressed candidates skip the actual IoU calculation, but
the rank scan itself still executes. This is the principal latency stress case.

The additional `test_two_class_6300()` function is currently disabled in
`main()`. Its boxes overlap but remain below the 0.5 IoU threshold, so it
selects all 6,300 boxes in each class and produces 12,600 output triplets.

## Current RTL co-simulation latency

The latest co-simulation completed on July 16, 2026 with Verilog RTL and passed
all four active tests.

| Transaction | Test | Latency (cycles) | Nominal time at 3.3 ns |
|---:|---|---:|---:|
| 0 | Corner format | 2,455 | 8.102 us |
| 1 | Center format | 4,541 | 14.985 us |
| 2 | Case A | 40,098 | 132.323 us |
| 3 | Case B | 302,425 | 998.003 us |

Summary:

| Metric | Cycles |
|---|---:|
| Minimum latency | 2,455 |
| Average latency | 87,379 |
| Maximum latency | 302,425 |
| Total execution time | 317,498 |

The time conversions use the nominal 3.3 ns target clock. Actual deployed time
depends on the final implementation frequency.

The source reports are:

```text
nms_onnx_component/hls/sim/report/nms_onnx_cosim.rpt
nms_onnx_component/hls/sim/report/verilog/result.transaction.rpt
```

## Latency optimization history

The following table records the measured RTL co-simulation latency through the
main experiments:

| Experiment | Corner | Center | Case A | Case B | Total |
|---|---:|---:|---:|---:|---:|
| Sorted-rank suppression state | 8,739 | 17,117 | 50,225 | 522,208 | 543,560 |
| Rank-banked sorted coordinates, eight IoU engines | 8,746 | 17,127 | 46,386 | 302,425 | 323,784 |
| Suppression initialization fused into gather | 2,454 | 4,541 | 40,094 | 302,425 | 317,492 |
| BF16 ports, FP32 internal processing | 2,455 | 4,541 | 40,098 | 302,425 | 317,498 |

Changing suppression state from original-box indexing to sorted-rank indexing
reduced the synthesized IoU-loop II from 136 to 4. The first complete
transaction-level latency baseline retained in this experiment is the
sorted-rank run shown above. Gathering coordinates into rank-banked arrays then
enabled eight IoU engines at II=1 and reduced Case B by 42.1% relative to that
baseline.

Fusing suppression initialization primarily helps workloads where `num_cand` is
much smaller than `MAX_BOXES`, such as Case A. It does not reduce Case B because
the full-length coordinate gather was already on the same critical path.

Changing only the ports to BF16 halves the input representation but does not
reduce Case B cycles because the values are immediately expanded to FP32 and
the compute architecture is unchanged.

## Current synthesis summary

The latest C-synthesis estimate reports:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM18K | 405 | 1,934 | 20% |
| DSP | 105 | 1,968 | 5% |
| FF | 27,448 | 1,799,680 | 1% |
| LUT | 30,275 | 899,840 | 3% |
| URAM | 4 | 463 | <1% |

The eight-lane IoU module uses 88 DSPs. The remaining DSPs are primarily in
input conversion and center-format handling.

The current top-level estimated period is 3.916 ns against the 3.3 ns target.
The limiting synthesis path is associated with radix histogram
read-modify-write logic rather than the II=1 IoU scan. Implementation timing
must therefore be checked before treating 3.3 ns as the achieved hardware
period.

## Running Vitis HLS

Configure the tool environment with Vitis/Vivado v2026.1

Run all stages:

```bash
vitis-run --mode hls --tcl run.tcl
```

Run selected stages:

```bash
HLS_STEPS="csim" vitis-run --mode hls --tcl run.tcl
HLS_STEPS="csim csynth" vitis-run --mode hls --tcl run.tcl
HLS_STEPS="cosim impl" vitis-run --mode hls --tcl run.tcl
```

## Standalone C++ validation

`ap_float` simulation requires the Vitis floating-point C model and its bundled
GMP/MPFR libraries:

FPO_LIB="$XILINX_HLS/functional_suite/data/ip/xfft_v9_1"

g++ -std=gnu++17 -O2 \
    -DMAX_BOXES=6300 \
    -DNUM_CLASSES=2 \
    -Wno-unknown-pragmas \
    -I"$XILINX_HLS/include" \
    rearchitect/v1/src/kernel_hls.cpp \
    rearchitect/v1/tb/testbench.cpp \
    -L"$FPO_LIB" \
    -lIp_floating_point_v7_1_bitacc_cmodel \
    -Wl,-l:libmpfr.so.4 \
    -Wl,-l:libgmp.so.11 \
    -o /tmp/nms_test

LD_LIBRARY_PATH="$FPO_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    /tmp/nms_test
```

## Potential next optimizations

- Build radix keys while filtering to remove the separate key-build pass.
- Keep scores as 16-bit BF16 keys internally and use two radix passes instead
  of four.
- Process the two classes concurrently with private class scratch buffers.
- Use fixed-point geometry to reduce IoU DSP and BRAM usage.
- Pack BF16 input values into wider AXI words, especially for `scores_in`,
  which currently remains a 16-bit AXI data path.
