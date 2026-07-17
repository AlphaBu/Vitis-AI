# ML VART + PL (NMS) Application — Developer Guide

<!--
## Copyright and license statement

Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
-->

Note: Example model names, JSON files, IP addresses, and commands are for reference only. Modify them for your compiled models and board.

This document is a **step-by-step guide** to writing a C++ application that:

1. runs a Vitis AI compiled model with the **VART-ML** API (`vart::Runner`), and
2. post-processes the inference **output tensor** with an **ONNX NonMaxSuppression (NMS) PL (Programmable Logic) kernel** (`nms_onnx`) using the native **XRT C++ API**, writing the kernel's **selected detections** as the final application output.

It then shows how to **build** the application with the Vitis AI SDK and **run it on a VEK385 board**, with copy-paste example commands.

The reference implementation for everything below is [`ml_vart_plus_pl.cpp`](ml_vart_plus_pl.cpp). It is derived from the plain-inference [`ml_vart`](../ml_vart) example; the *only* additions are the NMS-post-processing pieces described in [Part 1](#part-1-writing-the-code). If you already have a working VART-ML inference app, you can add NMS post-processing by copying just those pieces.

The example uses the `nms_onnx` kernel, a real hardware NMS operator. Unlike a byte-for-byte identity copy, NMS **transforms** the data: it takes candidate boxes and per-class scores and returns only the surviving detections. Because of that, a small **host-side data-conversion step** is required between the model output and the kernel (the model output is not already in the kernel's boxes/scores layout). The full **host → `xrt::bo` → PL kernel → `xrt::bo` → host** data path is otherwise the same as any real PL post-processing kernel would use.

---

## Table of Contents

- [Concept](#concept)
- [Prerequisites](#prerequisites)
- [Part 1: Writing the code](#part-1-writing-the-code)
  - [1.1 Includes and linking](#11-includes-and-linking)
  - [1.2 Understand the PL kernel contract](#12-understand-the-pl-kernel-contract)
  - [1.3 Write a thin XRT wrapper for the kernel](#13-write-a-thin-xrt-wrapper-for-the-kernel)
  - [1.4 Initialize the kernel once](#14-initialize-the-kernel-once)
  - [1.5 Convert and forward the output after each inference](#15-convert-and-forward-the-output-after-each-inference)
  - [1.5.1 The data-conversion step (model output → boxes/scores)](#151-the-data-conversion-step-model-output--boxesscores)
  - [1.6 Wire it into `main()`](#16-wire-it-into-main)
  - [1.7 Extend the JSON config](#17-extend-the-json-config)
- [Part 2: Building](#part-2-building)
- [Part 3: Running on the board](#part-3-running-on-the-board)
  - [3.1 Board prerequisites](#31-board-prerequisites)
  - [3.2 Get the files onto the board](#32-get-the-files-onto-the-board)
  - [3.3 Set the runtime environment](#33-set-the-runtime-environment)
  - [3.4 Pick the correct PL device index](#34-pick-the-correct-pl-device-index)
  - [3.5 Example run commands](#35-example-run-commands)
  - [3.6 Verify correctness](#36-verify-correctness)
- [Command-line arguments](#command-line-arguments)
- [Input / output file layout](#input--output-file-layout)
- [Troubleshooting](#troubleshooting)
- [Related documents](#related-documents)

---

## Concept

```
                VART-ML (NPU/AIE)              host convert         XRT native API (PL)
  IFM .bin ──► vart::Runner::run() ──► OFM tensor ──► boxes+scores ──► nms_onnx kernel ──► detections .bin
              (device 0, amdxdna)                                     (device 1, PL region)
```

For every inference call the app:

1. reads input frames into the runner's input `vart::NpuTensor`s,
2. runs `vart::Runner`,
3. **converts** each output tensor from the model's per-anchor layout into the kernel's separate `boxes` and `scores` arrays (host side),
4. copies those arrays into `xrt::bo`s, launches the `nms_onnx` kernel, and reads the selected detections back,
5. writes the **NMS-selected detections** (not the raw runner output) to disk.

Unlike an identity copy, NMS is **data-transforming**: the output is a variable-length list of surviving detections, not a reshaped copy of the input. The data-conversion in step 3 is a distinct, separately timed stage (see [1.5.1](#151-the-data-conversion-step-model-output--boxesscores)).

---

## Prerequisites

To **build** you need:

- The Vitis AI SDK for Versal AI Edge Series Gen 2 (provides the cross toolchain, `vart-ml`, and `xrt` via `pkg-config`).

To **run on the board** you need:

- A VEK385 board booted with the platform image, network reachable.
- A compiled model artifact (`.rai`) or model cache directory, and matching input `.bin` files.
- An `.xclbin` on the board that contains the `nms_onnx` PL kernel. On the VEK385 reference design this is the combined `x_plus_ml.xclbin` (which also carries the AIE/ML overlay). See the [`nms_hls` kernel](../../../reference_design/vek385/rev-a_nms/nms_hls) and the VEK385 reference-design Vitis flow for building the `.xclbin`.

---

## Part 1: Writing the code

The following mirrors what `ml_vart_plus_pl.cpp` does, in the order you would add it to a VART-ML app.

### 1.1 Includes and linking

Add the native XRT headers on top of your existing VART-ML includes:

```cpp
#include <vart/vart_runner_factory.hpp>   // VART-ML (already present in ml_vart)

// Native XRT API for the PL kernel:
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>
#include <xrt/experimental/xrt_xclbin.h>   // xrt::xclbin (for register_xclbin)

#include <cstring>   // std::memcpy
#include <memory>    // std::unique_ptr
```

Link against `xrt` in addition to `vart-ml` (see [Part 2](#part-2-building) for the Makefile changes).

### 1.2 Understand the PL kernel contract

The `nms_onnx` HLS kernel implements the ONNX NonMaxSuppression operator. Its signature is:

```cpp
void nms_onnx(const bf16_t *boxes_in,      // gmem0, arg0
              const bf16_t *scores_in,     // gmem1, arg1
              int32_t      *selected_out,  // gmem2, arg2
              int  num_batches,            // arg3
              int  num_classes,            // arg4
              int  num_boxes,              // arg5
              int  max_out_per_class,      // arg6
              bf16_t iou_threshold,        // arg7 (16-bit s_axilite scalar)
              bf16_t score_threshold,      // arg8 (16-bit s_axilite scalar)
              int  center_point_box,       // arg9
              int32_t *num_selected);      // gmem3, arg10
```

- **Data ports are `bf16`** (16-bit; the high 16 bits of an IEEE-754 float32).
- **Input layouts:**
  - `boxes_in [num_batches * num_boxes * 4]` — per box `[cx, cy, w, h]` when `center_point_box = 1` (YOLOX), or corners when `0`.
  - `scores_in [num_batches * num_classes * num_boxes]` — **class-major**.
- **Output layouts (`int32`):**
  - `selected_out [num_selected * 3]` — triplets `[batch_idx, class_idx, box_idx]`.
  - `num_selected[0]` — the number of valid triplets.
- **Scalar arguments** (`num_batches`, `num_classes`, `num_boxes`, `max_out_per_class`, `center_point_box`) are plain `int`s; the two thresholds are passed as their **16-bit bf16 bit patterns**.
- Argument index → memory bank mapping (`group_id(0)` = gmem0, … `group_id(10)` = gmem3) is what you pass to `xrt::bo` so each buffer is allocated in the bank the kernel expects.

When you write host code you must respect *this* kernel's argument order, layouts, and units. A different kernel will have a different signature — adjust the `xrt::kernel(...)` call and the `m_kernel(...)` launch accordingly.

### 1.3 Write a thin XRT wrapper for the kernel

Encapsulate device/xclbin/kernel setup, the persistent input/output buffers, and a single `run()` operation. This is the `pl_nms` class in the reference. First, two bf16 helpers used both here and in the conversion step:

```cpp
// bf16 bit pattern -> float (bf16 is the high 16 bits of a float32).
static inline float bf16_bits_to_float(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// float -> bf16 bit pattern, round-to-nearest-even.
static inline uint16_t float_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t rounding_bias = 0x00007FFFu + ((u >> 16) & 1u);
  u += rounding_bias;
  return static_cast<uint16_t>(u >> 16);
}
```

The wrapper:

```cpp
class pl_nms {
 public:
  pl_nms(const std::string& xclbin_path,
         const std::string& kernel_name,
         AppLogLevel app_log,
         unsigned int device_index = 0)
      : m_app_log(app_log) {
    // 1. Open the XRT device that hosts the PL region (see section 3.4).
    m_device = xrt::device(device_index);

    // 2. Register the xclbin (modern, non-deprecated path — do NOT use
    //    device.load_xclbin(), it is deprecated).
    xrt::xclbin xclbin(xclbin_path);
    xrt::uuid uuid = m_device.register_xclbin(xclbin);

    // 3. A hw_context binds the registered xclbin; the kernel is created from it.
    xrt::hw_context ctx(m_device, uuid);
    m_kernel = xrt::kernel(ctx, kernel_name);
  }

  // Allocate the persistent NMS input/output buffers and cache the scalar
  // arguments. Call once, after the output geometry is known (section 1.4).
  void configure(int num_batches, int num_classes, int num_boxes,
                 int max_out_per_class, float iou_threshold,
                 float score_threshold, int center_point_box) {
    m_num_batches = num_batches;   m_num_classes = num_classes;
    m_num_boxes = num_boxes;       m_max_out_per_class = max_out_per_class;
    m_center_point_box = center_point_box;
    m_iou_bits   = float_to_bf16_bits(iou_threshold);    // thresholds -> bf16 bits
    m_score_bits = float_to_bf16_bits(score_threshold);

    const size_t n_boxes  = size_t(num_batches) * num_boxes * 4;             // [cx,cy,w,h]
    const size_t n_scores = size_t(num_batches) * num_classes * num_boxes;   // class-major
    m_selected_capacity   = size_t(num_batches) * num_classes * num_boxes;   // upper bound

    m_boxes_bo    = xrt::bo(m_device, n_boxes  * sizeof(uint16_t), m_kernel.group_id(0));   // gmem0
    m_scores_bo   = xrt::bo(m_device, n_scores * sizeof(uint16_t), m_kernel.group_id(1));   // gmem1
    m_selected_bo = xrt::bo(m_device, m_selected_capacity * 3 * sizeof(int32_t),
                            m_kernel.group_id(2));                                          // gmem2
    m_numsel_bo   = xrt::bo(m_device, sizeof(int32_t), m_kernel.group_id(10));              // gmem3

    m_boxes_host    = m_boxes_bo.map<uint16_t*>();     // caller fills these two
    m_scores_host   = m_scores_bo.map<uint16_t*>();
    m_selected_host = m_selected_bo.map<int32_t*>();   // results read from here
    m_numsel_host   = m_numsel_bo.map<int32_t*>();
  }

  // Host mappings the caller fills before run() (bf16 bit patterns):
  //   boxes  [num_boxes*4]           layout [cx,cy,w,h]
  //   scores [num_classes*num_boxes] class-major
  uint16_t* boxes_host()  { return m_boxes_host; }
  uint16_t* scores_host() { return m_scores_host; }

  // Sync inputs -> device, launch the kernel, wait, sync results -> host.
  void run() {
    m_boxes_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    m_scores_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Argument order matches the kernel signature; thresholds are bf16 bit patterns.
    auto run = m_kernel(m_boxes_bo, m_scores_bo, m_selected_bo, m_num_batches,
                        m_num_classes, m_num_boxes, m_max_out_per_class,
                        m_iou_bits, m_score_bits, m_center_point_box, m_numsel_bo);
    run.wait();

    m_numsel_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    m_selected_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  // Results of the last run(): count and the selected triplets [batch,class,box].
  int            num_selected()      const { return m_numsel_host[0]; }
  const int32_t* selected()          const { return m_selected_host; }
  size_t         selected_capacity() const { return m_selected_capacity; }

 private:
  xrt::device m_device;
  xrt::kernel m_kernel;
  xrt::bo     m_boxes_bo, m_scores_bo, m_selected_bo, m_numsel_bo;
  uint16_t   *m_boxes_host = nullptr, *m_scores_host = nullptr;
  int32_t    *m_selected_host = nullptr, *m_numsel_host = nullptr;
  int    m_num_batches = 1, m_num_classes = 0, m_num_boxes = 0;
  int    m_max_out_per_class = 0, m_center_point_box = 1;
  uint16_t m_iou_bits = 0, m_score_bits = 0;
  size_t m_selected_capacity = 0;
  AppLogLevel m_app_log;
};
```

Key points:

- **Use `register_xclbin` + `hw_context`, not `device.load_xclbin()`** — the latter is deprecated and will fail a `-Wdeprecated`/`-Werror` build.
- **Buffers are allocated once** in `configure()` (sized to the model's `num_boxes` / `num_classes`) and reused for every frame; `selected_out` is sized to the worst-case capacity `num_batches*num_classes*num_boxes`.
- **`sync()` in both directions** is mandatory: `BO_TO_DEVICE` for the two inputs before the launch, `BO_FROM_DEVICE` for the count and selected triplets after.
- The two thresholds are **passed as bf16 bit patterns** (16-bit `s_axilite` scalars), so they are converted with `float_to_bf16_bits` in `configure()`.

> The reference `pl_nms` also keeps per-stage timers (host→PL sync, kernel exec, PL→host sync) so `--benchmark` can report each separately; they are omitted above for brevity.

### 1.4 Initialize the kernel once

Create the wrapper once, after the runner and tensors are allocated. The NMS geometry is derived from the **model output tensor**: its last dimension is the per-anchor `stride` (`[cx,cy,w,h,obj,cls0,cls1,…]`), and `num_boxes = (total elements) / stride`. `num_classes`, `max_out_per_class`, and the thresholds come from the JSON config. Skip it in `--dry-run` (there is no real data to forward):

```cpp
std::unique_ptr<pl_nms> m_pl = nullptr;   // member of your app context

vart_app_status init_pl_kernel() {
  if (m_app_opt.dry_run) return vart_app_status::SUCCESS;   // nothing to forward
  if (m_app_opt.pl_xclbin.empty()) { /* error: xclbin-location required */ }
  if (!fs::exists(m_app_opt.pl_xclbin)) { /* error: file not found */ }
  if (m_output_tensors_info.empty()) { /* error: no output tensor to drive NMS */ }

  // Derive NMS geometry from the single model output tensor.
  const vart::NpuTensorInfo& oi = m_output_tensors_info[0];
  size_t total = 1;
  for (uint32_t d : oi.shape) total *= d;
  m_nms_stride = oi.shape.empty() ? 0 : static_cast<int>(oi.shape.back());
  if (m_nms_stride <= 0 || total % size_t(m_nms_stride) != 0) { /* error: bad geometry */ }
  m_nms_num_boxes   = static_cast<int>(total / size_t(m_nms_stride));
  m_nms_num_classes = m_app_opt.nms_num_classes;
  // The stride must hold 4 box coords + 1 objectness + num_classes class scores.
  if (m_nms_stride < 5 + m_nms_num_classes) { /* error: stride too small */ }

  try {
    m_pl = std::make_unique<pl_nms>(m_app_opt.pl_xclbin, m_app_opt.pl_kernel,
                                    m_app_opt.app_log, m_app_opt.pl_device_index);
    m_pl->configure(/*num_batches=*/1, m_nms_num_classes, m_nms_num_boxes,
                    m_app_opt.nms_max_out_per_class, m_app_opt.nms_iou_threshold,
                    m_app_opt.nms_score_threshold, m_app_opt.nms_center_point_box);
  } catch (const std::exception& e) { /* log e.what(); return FAILURE */ }

  m_nms_counts.assign(m_batch_size, 0);        // per-frame result storage
  m_nms_results.assign(m_batch_size, {});
  return vart_app_status::SUCCESS;
}
```

### 1.5 Convert and forward the output after each inference

For every valid frame: (1) convert the model output into the kernel's `boxes`/`scores` layout (host side), (2) run the kernel, and (3) capture the selected detections:

```cpp
vart_app_status forward_outputs_through_pl(size_t actual_batch_size) {
  for (size_t i = 0; i < actual_batch_size; ++i) {
    // Stage 1: host-side data conversion (model output -> NMS layout).
    if (!convert_model_output_to_nms_input(i)) return vart_app_status::FAILURE;

    // Stage 2: run the NMS PL kernel.
    m_pl->run();

    // Stage 3: capture this frame's detections (clamp to the output capacity).
    const int count = m_pl->num_selected();
    const size_t cap = m_pl->selected_capacity();
    const size_t take = std::min(size_t(std::max(count, 0)), cap);
    m_nms_counts[i] = static_cast<int32_t>(take);
    m_nms_results[i].assign(m_pl->selected(), m_pl->selected() + take * 3);
  }
  return vart_app_status::SUCCESS;
}
```

`m_nms_counts[i]` is the number of detections for frame `i`; `m_nms_results[i]` holds `take*3` `int32` values (the `[batch,class,box]` triplets), used later by `write_output_tensors`.

### 1.5.1 The data-conversion step (model output → boxes/scores)

The NMS kernel does **not** consume the model output directly. A YOLOX-style model emits one tensor of shape `[.., num_boxes, stride]` where each anchor is `[cx, cy, w, h, obj, cls0, cls1, …(padding)]`, but `nms_onnx` expects **two separate** bf16 arrays — `boxes [num_boxes*4]` and class-major `scores [num_classes*num_boxes]`. This mismatch is exactly why a byte-for-byte / zero-copy forward is **not** possible here (in contrast to an identity kernel): the host must reshape and combine the data first.

Two things happen per anchor:

1. **Box coords** `[cx,cy,w,h]` are copied verbatim as bf16 bit patterns.
2. **Score** = `objectness × class_probability` (the YOLOX convention). Neither the NPU model nor the NMS kernel performs this multiply, so it is done here, and the result is written class-major.

```cpp
bool convert_model_output_to_nms_input(size_t i) {
  const uint16_t* mo = static_cast<const uint16_t*>(m_outputs[i][0].get_virtual_address());
  if (!mo) return false;
  uint16_t* boxes  = m_pl->boxes_host();
  uint16_t* scores = m_pl->scores_host();
  const int stride = m_nms_stride, num_boxes = m_nms_num_boxes, num_classes = m_nms_num_classes;

  for (int n = 0; n < num_boxes; ++n) {
    const uint16_t* a = mo + size_t(n) * stride;
    // Box coords [cx,cy,w,h]: pure bf16 bit-pattern copy.
    boxes[size_t(n) * 4 + 0] = a[0];
    boxes[size_t(n) * 4 + 1] = a[1];
    boxes[size_t(n) * 4 + 2] = a[2];
    boxes[size_t(n) * 4 + 3] = a[3];
    // Confidence = objectness * class probability (class-major layout).
    const float obj = bf16_bits_to_float(a[4]);
    for (int c = 0; c < num_classes; ++c) {
      const float conf = obj * bf16_bits_to_float(a[5 + c]);
      scores[size_t(c) * num_boxes + n] = float_to_bf16_bits(conf);
    }
  }
  return true;
}
```

This conversion is a **distinct pipeline stage**, timed separately from the PL transfers and the kernel execution so `--benchmark` can report it on its own line (see [3.5](#35-example-run-commands)).

### 1.6 Wire it into `main()`

Two changes to the plain-inference flow:

**(a)** Initialize the kernel after the tensors are allocated:

```cpp
runner->allocate_input_tensors();
runner->allocate_output_tensors();
if (vart_app_status::FAILURE == runner->init_pl_kernel()) return -1;   // <-- added
// ... then run_inference_and_save(...) as usual
```

**(b)** Convert+forward before writing, in the normal-inference path only. Benchmark/dry-run `continue` before writing (benchmark still runs the PL forward to time it, but skips file I/O):

```cpp
run_inference();                       // vart::Runner produces the outputs

if (options.dry_run) { /* ... */ continue; }         // no PL, no I/O

forward_outputs_through_pl(actual_batch_size);        // <-- convert + NMS kernel

if (benchmark) continue;                              // timed, but skip writing
write_output_tensors(frame_count, actual_batch_size); // writes the NMS detections
```

That's the whole integration: **init once, convert+forward, then write.**

### 1.7 Extend the JSON config

Add a `pl-config` object to the app-config JSON that also carries the NMS operator parameters:

```json
"pl-config": {
  "xclbin-location": "/run/media/mmcblk0p1/x_plus_ml.xclbin",
  "kernel-name": "nms_onnx",
  "device-index": 1,
  "num-classes": 2,
  "max-output-boxes-per-class": 200,
  "iou-threshold": 0.65,
  "score-threshold": 0.01,
  "center-point-box": 1
}
```

Parse it (Boost.PropertyTree) after the existing fields, and require `xclbin-location` unless `--dry-run`:

```cpp
if (auto pl_config = config.get_child_optional("pl-config")) {
  const auto& pl = pl_config.get();
  app_info.pl_xclbin           = pl.get<std::string>("xclbin-location", app_info.pl_xclbin);
  app_info.pl_kernel           = pl.get<std::string>("kernel-name",     app_info.pl_kernel);        // default "nms_onnx"
  app_info.pl_device_index     = pl.get<unsigned int>("device-index",   app_info.pl_device_index);  // default 1
  app_info.nms_num_classes     = pl.get<int>("num-classes",                app_info.nms_num_classes);
  app_info.nms_max_out_per_class = pl.get<int>("max-output-boxes-per-class", app_info.nms_max_out_per_class);
  app_info.nms_iou_threshold   = pl.get<float>("iou-threshold",   app_info.nms_iou_threshold);
  app_info.nms_score_threshold = pl.get<float>("score-threshold", app_info.nms_score_threshold);
  app_info.nms_center_point_box = pl.get<int>("center-point-box", app_info.nms_center_point_box);
}
if (!app_info.dry_run && app_info.pl_xclbin.empty()) { /* error: xclbin-location required */ }
```

The full schema is documented in [json_configs/README.md](json_configs/README.md).

---

## Part 2: Building

The application links against both `vart-ml` and `xrt` via `pkg-config`. Relative to the base `ml_vart` Makefile, add the `xrt` flags and dependency check:

```make
check_dependencies:
	@pkg-config --atleast-version=0.1 vart-ml || (echo "vart-ml >= 0.1 required"; exit 1)
	@pkg-config --exists xrt || (echo "xrt is required for the nms_onnx PL kernel"; exit 1)

CFLAGS  += `pkg-config --cflags vart-ml` `pkg-config --cflags xrt` -I../common/include -std=c++17
LDFLAGS += `pkg-config --libs vart-ml`   `pkg-config --libs xrt`   -lboost_program_options
```

Then:

1. Source the Vitis AI SDK for Versal AI Edge Series Gen 2:

   ```bash
   source /path/to/sdk/environment-setup-cortexa72-cortexa53-amd-linux
   ```

2. Build:

   ```bash
   make all
   ```

   The resulting binary is `ml_vart_plus_pl` (an AArch64 ELF).

3. Clean:

   ```bash
   make clean
   ```

---

## Part 3: Running on the board

### 3.1 Board prerequisites

Before running, finish board setup for your platform: program the required PL + AIE overlay, and make the `nms_onnx` `.xclbin` available on the board at the path you will reference in `pl-config.xclbin-location` (e.g. `/run/media/mmcblk0p1/x_plus_ml.xclbin`).

### 3.2 Get the files onto the board

You need four things on the board:

| Item                         | Example board path                                             |
| ---------------------------- | ------------------------------------------------------------- |
| The `ml_vart_plus_pl` binary | `/home/root/ml_vart_plus_pl`                                  |
| The compiled model (`.rai`)  | `<design>/yolox_nano_onnx_pt_regular_conv_all/....rai`        |
| The input `.bin` file(s)     | `<design>/inputSubBO_..._trim.bin`                            |
| The app-config JSON          | `<design>/vart_config_plus_pl.json`                           |

Copy with `scp` (adjust IP/paths):

```bash
scp ml_vart_plus_pl root@10.25.38.208:/home/root/
scp -r <design_dir> root@10.25.38.208:/home/root/
```

Tip: if your build tree is exported to the board over NFS, you can run the binary directly from the mount point instead of copying (e.g. `/mnt/.../ml_vart_plus_pl`).

### 3.3 Set the runtime environment

VART-ML loads `libflexmlrt.so` at runtime, which is **not** on the default library path. Export it before running (adjust the path to your image):

```bash
export LD_LIBRARY_PATH=/usr/lib/python3.12/site-packages/flexmlrt/lib:$LD_LIBRARY_PATH
```

Without this you will see:

```
ERROR: Cannot open library libflexmlrt.so: ... cannot open shared object file
```

### 3.4 Pick the correct PL device index

On VEK385 there are **two XRT devices**: the **NPU/AIE is device `0`** and the **PL region is device `1`**. The PL kernel therefore lives on **device 1**, which is why `pl-config.device-index` defaults to `1`. If you point the wrapper at device `0` the kernel will not be found. Set `device-index` in the JSON if your platform differs.

### 3.5 Example run commands

All commands below assume you `cd` into the design directory (so the relative `model-file` / `ifms-config.file` paths resolve) and have exported `LD_LIBRARY_PATH` as in [3.3](#33-set-the-runtime-environment).

- **Normal inference** (default HW-tensor mode; runs NMS and writes the selected detections):

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json
  ```

- **With INFO logging** — shows the PL device open / xclbin load / kernel-ready lines, useful for first-run debugging:

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --log-level 5
  ```

  Expected PL lines:

  ```
  Opening XRT device 1 for PL kernel 'nms_onnx'
  Loading xclbin: /run/media/mmcblk0p1/x_plus_ml.xclbin
  PL kernel 'nms_onnx' ready
  NMS configured: num_boxes=... stride=... num_classes=... max_out_per_class=...
  ```

- **Dry run** — validate the config and model without any file I/O or PL forwarding (PL init is skipped):

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --dry-run --log-level 5
  ```

- **Benchmark** for 100 runs — times the full datapath but saves no outputs. It reports the overall ML-only average plus a per-stage breakdown (ms/frame): NPU **ML inference**, **data-conversion (host)** (model output → boxes/scores), **data-transfer-to-PL** (host→PL input sync), the **NMS PL kernel** (launch + wait), and **data-transfer-from-PL** (PL→host result sync). The `total (end-to-end)` line is the sum of the five stages:

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --benchmark --runs 100
  ```

  Example output:

  ```
  Average inference time over 100 runs (ML only): 1.76 ms
  Per-stage average (ms/frame):
    ML inference             : 1.765
    data-conversion (host)   : 0.691
    data-transfer-to-PL      : 0.004
    NMS PL kernel            : 0.222
    data-transfer-from-PL    : 0.006
    ------------------------------------
    total (end-to-end)       : 2.688
  Run completed successfully.
  ```

  The NMS inputs (`boxes` + `scores`) are small relative to a full OFM, so both host↔PL transfers are near-negligible; the host-side data conversion is the largest post-processing cost.

- **Inspect model metadata** (no inference) — prints the CPU + HW tensor view and dumps `<model_basename>_info.json`. Use it to look up the input tensor `name`s needed for `ifms-config` and to confirm the output stride:

  ```bash
  ml_vart_plus_pl --get-model-info yolox_nano_onnx_pt_regular_conv_all/yolox_nano_onnx_pt_regular_conv_all.rai
  ```

### 3.6 Verify correctness

Because `nms_onnx` is a **data-transforming** kernel (not an identity copy), its output is **not** byte-comparable to a plain `ml_vart` run. Verify instead against a reference NMS implementation: run the model output through a CPU/ONNX NonMaxSuppression with the **same** parameters (`num_classes`, `max_output_boxes_per_class`, `iou_threshold`, `score_threshold`, `center_point_box`) and confirm the selected `[batch, class, box]` triplets match (allowing for ordering and bf16 rounding). For an end-to-end sanity check, decode the selected boxes and compare detections (or mAP) against your reference pipeline.

---

## Command-line arguments

| Option              | Required  | Default | Description                                                  |
| ------------------- | --------- | ------- | ------------------------------------------------------------ |
| `--app-config`      | Mandatory |         | Path to configuration JSON file. Mandatory for inference / dry-run / benchmark flows. Ignored when `--get-model-info` is supplied. |
| `--runs`            | Optional  | `1`     | Number of iterations to run. |
| `--benchmark`       | Optional  | `false` | Benchmark the model for `n` runs. Times the full ML→convert→NMS datapath and prints a per-stage breakdown; skips output saving. |
| `--log-level`       | Optional  | `2`     | Log level: `1`=ERROR, `2`=WARNING, `5`=INFO, `6`=DEBUG. |
| `--frames`          | Optional  | all     | Number of frames to process per iteration. |
| `--dry-run`         | Optional  |         | Skip file I/O and PL kernel initialization (test configuration only). |
| `--get-model-info <model-path>` | Optional |  | Standalone mode: print the model's CPU + HW tensor metadata and dump `<model_basename>_info.json`; no inference. `--app-config` is ignored. |
| `--help`            | Optional  |         | Print help and exit. |

---

## Input / output file layout

**Input** — one `.bin` per input tensor, frames concatenated back-to-back in the layout `vart::Runner` expects. Each file size must be an exact multiple of that tensor's per-frame `size_in_bytes` (query with `--get-model-info`). The same frame index across all tensor files forms one batch element; the batch size `N` is fixed by the compiled model. Partial final batches run with fewer frames (no padding).

**Output** — a **single** binary file, `nms_selected.bin`, holding the NMS-selected detections for every frame written sequentially (each frame's result is variable-length, so fixed offsets cannot be used). Per frame:

```
int32                num_selected
num_selected * 3 * int32   triplets [batch_idx, class_idx, box_idx]
```

- Single run: `nms_selected.bin`.
- Multiple runs (`--runs > 1`): `iter{run_index}_nms_selected.bin` (`{run_index}` 0-based).

The `box_idx` in each triplet indexes back into the model's anchor list, so the corresponding box coordinates can be recovered from the model output tensor if you need decoded detections.

---

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `Cannot open library libflexmlrt.so` | `LD_LIBRARY_PATH` missing the flexmlrt lib dir — see [3.3](#33-set-the-runtime-environment). |
| Kernel/xclbin load fails, or `xrt::kernel` throws "no such kernel" | Wrong `device-index` (use `1` on VEK385, see [3.4](#34-pick-the-correct-pl-device-index)), wrong `kernel-name`, or the `.xclbin` does not contain the kernel. |
| `xclbin not found` | `pl-config.xclbin-location` path does not exist on the board. |
| Build error: `load_xclbin` is deprecated | Use the `register_xclbin` + `hw_context` path (see [1.3](#13-write-a-thin-xrt-wrapper-for-the-kernel)). |
| `Model output stride too small` / `Unexpected model output geometry` | The model output last dimension (stride) must be at least `5 + num-classes` (4 box coords + objectness + class scores). Check `num-classes` in `pl-config` against the model. |
| Zero or too few detections | Thresholds too strict, or the objectness×class score multiply / class-major layout not matching your model. Verify `score-threshold` / `iou-threshold` and the conversion in [1.5.1](#151-the-data-conversion-step-model-output--boxesscores). |

---

## Related documents

- [json_configs/README.md](json_configs/README.md) — full JSON schema for the app-config (`inference-config`, `ifms-config`, `ofms-dir`, `pl-config`).
- [../ml_vart](../ml_vart) — the plain-inference base example this one extends.
- [../../../reference_design/vek385/rev-a_nms/nms_hls](../../../reference_design/vek385/rev-a_nms/nms_hls) — the `nms_onnx` HLS PL kernel source and Vitis build flow.
- [../../docs/runner_options.md](../../../docs/runner_options.md) — the full `runner-options` schema.
