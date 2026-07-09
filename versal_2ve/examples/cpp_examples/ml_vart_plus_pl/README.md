# ML VART + PL (pass_through) Application — Developer Guide

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
2. forwards each inference **output tensor** through a **PL (Programmable Logic) kernel** using the native **XRT C++ API**, using the PL kernel's output as the final application output.

It then shows how to **build** the application with the Vitis AI SDK and **run it on a VEK385 board**, with copy-paste example commands.

The reference implementation for everything below is [`ml_vart_plus_pl.cpp`](ml_vart_plus_pl.cpp). It is derived from the plain-inference [`ml_vart`](../ml_vart) example; the *only* additions are the PL-forwarding pieces described in [Part 1](#part-1-writing-the-code). If you already have a working VART-ML inference app, you can add PL post-processing by copying just those pieces.

The example uses the `pass_through` kernel, which is an **identity copy** kernel (it copies its input words to its output unchanged). This keeps the example easy to verify — the saved outputs are byte-for-byte identical to a plain `ml_vart` run — while exercising the full **host → `xrt::bo` → PL kernel → `xrt::bo` → host** data path that a *real* PL post-processing kernel (NMS, dequantize, softmax, resize, …) would use. To use a real kernel, swap the kernel name/arguments and replace the identity logic; the host plumbing stays the same.

---

## Table of Contents

- [Concept](#concept)
- [Prerequisites](#prerequisites)
- [Part 1: Writing the code](#part-1-writing-the-code)
  - [1.1 Includes and linking](#11-includes-and-linking)
  - [1.2 Understand the PL kernel contract](#12-understand-the-pl-kernel-contract)
  - [1.3 Write a thin XRT wrapper for the kernel](#13-write-a-thin-xrt-wrapper-for-the-kernel)
  - [1.4 Initialize the kernel once](#14-initialize-the-kernel-once)
  - [1.5 Forward the outputs after each inference](#15-forward-the-outputs-after-each-inference)
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
                VART-ML (NPU/AIE)                 XRT native API (PL)
  IFM .bin ──► vart::Runner::run() ──► OFM tensor ──► pass_through kernel ──► OFM .bin
              (device 0, amdxdna)                    (device 1, PL region)
```

For every inference call the app:

1. reads input frames into the runner's input `vart::NpuTensor`s,
2. runs `vart::Runner`,
3. copies each output tensor's bytes into an `xrt::bo`, launches the PL kernel, and reads the result back,
4. writes the **PL kernel output** (not the raw runner output) to disk.

The forwarding is **byte-oriented**: the byte count is rounded up to whole 32-bit words and the tail word is zero-padded, so it is **agnostic to the tensor data type** (int8 / bf16 / fp16 / fp32 / …) and works for any number of output tensors.

---

## Prerequisites

To **build** you need:

- The Vitis AI SDK for Versal AI Edge Series Gen 2 (provides the cross toolchain, `vart-ml`, and `xrt` via `pkg-config`).

To **run on the board** you need:

- A VEK385 board booted with the platform image, network reachable.
- A compiled model artifact (`.rai`) or model cache directory, and matching input `.bin` files.
- An `.xclbin` on the board that contains the `pass_through` PL kernel. On the VEK385 reference design this is the combined `x_plus_ml.xclbin` (which also carries the AIE/ML overlay). See the [`pass_through` kernel](../../../reference_design/vek385/rev-b/pass_through) and the VEK385 reference-design Vitis flow for building the `.xclbin`.

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

#include <cstring>   // std::memcpy / std::memset
#include <memory>    // std::unique_ptr
```

Link against `xrt` in addition to `vart-ml` (see [Part 2](#part-2-building) for the Makefile changes).

### 1.2 Understand the PL kernel contract

The `pass_through` HLS kernel has this signature:

```cpp
void pass_through(const ap_uint<32>* in, ap_uint<32>* out, int size);
```

- It copies `size` **32-bit words** from global-memory bank `gmem0` (argument 0, `in`) to `gmem1` (argument 1, `out`).
- `size` is a **word count**, not a byte count.
- Argument index → memory bank mapping (`group_id(0)` = gmem0, `group_id(1)` = gmem1) is what you pass to `xrt::bo` so each buffer is allocated in the bank the kernel expects.

When you write host code you must respect *this* kernel's argument order and units. A different kernel will have a different signature — adjust the `xrt::kernel(...)` call and the `m_kernel(...)` launch accordingly.

### 1.3 Write a thin XRT wrapper for the kernel

Encapsulate device/xclbin/kernel setup and a single `forward()` operation. This is the `pl_pass_through` class in the reference:

```cpp
class pl_pass_through {
 public:
  pl_pass_through(const std::string& xclbin_path,
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

  // Forward nbytes from src through the kernel, writing the result to dst.
  // src and dst may alias (in-place forward).
  void forward(const void* src, void* dst, size_t nbytes) {
    if (nbytes == 0) return;
    ensure_capacity(nbytes);

    const size_t nwords = (nbytes + 3) / 4;   // pass_through works on 32-bit words
    const size_t padded = nwords * 4;

    // Host -> input bo
    auto* in_host = m_in_bo.map<uint8_t*>();
    std::memcpy(in_host, src, nbytes);
    if (padded > nbytes) std::memset(in_host + nbytes, 0, padded - nbytes); // zero tail
    m_in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch: argument order matches the kernel signature (in, out, size).
    auto run = m_kernel(m_in_bo, m_out_bo, static_cast<int>(nwords));
    run.wait();

    // Output bo -> host
    m_out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(dst, m_out_bo.map<uint8_t*>(), nbytes);
  }

 private:
  // (Re)allocate device buffers to whole 32-bit words; reuse across calls.
  void ensure_capacity(size_t nbytes) {
    const size_t need = ((nbytes + 3) / 4) * 4;
    if (need <= m_capacity) return;
    m_in_bo  = xrt::bo(m_device, need, m_kernel.group_id(0));  // gmem0 (in)
    m_out_bo = xrt::bo(m_device, need, m_kernel.group_id(1));  // gmem1 (out)
    m_capacity = need;
  }

  xrt::device m_device;
  xrt::kernel m_kernel;
  xrt::bo     m_in_bo, m_out_bo;
  size_t      m_capacity = 0;
  AppLogLevel m_app_log;
};
```

Key points:

- **Use `register_xclbin` + `hw_context`, not `device.load_xclbin()`** — the latter is deprecated and will fail a `-Wdeprecated`/`-Werror` build.
- **Buffers are sized to whole 32-bit words** and reused; `ensure_capacity()` only reallocates when a bigger transfer is needed (the largest output tensor sets the high-water mark).
- **`sync()` in both directions** is mandatory around the launch (`BO_TO_DEVICE` before, `BO_FROM_DEVICE` after).
- **`forward()` supports aliasing** so you can write the result back over the same buffer (in-place).

### 1.4 Initialize the kernel once

Create the wrapper once, after the runner and tensors are allocated. Skip it in `--dry-run` (there is no real data to forward):

```cpp
std::unique_ptr<pl_pass_through> m_pl = nullptr;   // member of your app context

vart_app_status init_pl_kernel() {
  if (m_app_opt.dry_run) return vart_app_status::SUCCESS;          // nothing to forward
  if (m_app_opt.pl_xclbin.empty()) { /* error: xclbin-location required */ }
  if (!fs::exists(m_app_opt.pl_xclbin)) { /* error: file not found */ }
  try {
    m_pl = std::make_unique<pl_pass_through>(
        m_app_opt.pl_xclbin, m_app_opt.pl_kernel, m_app_opt.app_log,
        m_app_opt.pl_device_index);
  } catch (const std::exception& e) {
    /* log e.what(); return FAILURE */
  }
  return vart_app_status::SUCCESS;
}
```

### 1.5 Forward the outputs after each inference

For every valid frame and every output tensor, forward the runner result in place:

```cpp
vart_app_status forward_outputs_through_pl(size_t actual_batch_size) {
  for (size_t i = 0; i < actual_batch_size; ++i) {          // frames in this batch
    for (size_t j = 0; j < m_num_output_tensors; ++j) {     // output tensors
      const size_t nbytes = m_output_tensors_info[j].size_in_bytes;   // per-frame bytes
      void* data_ptr = m_outputs[i][j].get_virtual_address();         // NpuTensor buffer
      // Transfer -> run -> write result back over the same buffer (in-place).
      m_pl->forward(data_ptr, data_ptr, nbytes);
    }
  }
  return vart_app_status::SUCCESS;
}
```

`vart::NpuTensor::get_virtual_address()` returns the host-visible pointer to the tensor data; `NpuTensorInfo::size_in_bytes` is the **per-frame** size.

### 1.6 Wire it into `main()`

Two changes to the plain-inference flow:

**(a)** Initialize the kernel after the tensors are allocated:

```cpp
runner->allocate_input_tensors();
runner->allocate_output_tensors();
if (vart_app_status::FAILURE == runner->init_pl_kernel()) return -1;   // <-- added
// ... then run_inference_and_save(...) as usual
```

**(b)** Forward before writing, in the normal-inference path only. Benchmark/dry-run `continue` before writing, so they never touch the PL path:

```cpp
run_inference();                       // vart::Runner produces the outputs

if (options.dry_run || benchmark) {
  /* ... */ continue;                  // no file I/O, no PL forwarding
}

forward_outputs_through_pl(actual_batch_size);   // <-- added, in place
write_output_tensors(frame_count, actual_batch_size);   // writes the PL output
```

That's the whole integration: **init once, forward right before writing.**

### 1.7 Extend the JSON config

Add a `pl-config` object to the app-config JSON that the app parses:

```json
"pl-config": {
  "xclbin-location": "/run/media/mmcblk0p1/x_plus_ml.xclbin",
  "kernel-name": "pass_through",
  "device-index": 1
}
```

Parse it (Boost.PropertyTree) after the existing fields, and require `xclbin-location` unless `--dry-run`:

```cpp
if (auto pl_config = config.get_child_optional("pl-config")) {
  const auto& pl = pl_config.get();
  app_info.pl_xclbin       = pl.get<std::string>("xclbin-location", app_info.pl_xclbin);
  app_info.pl_kernel       = pl.get<std::string>("kernel-name",     app_info.pl_kernel);       // default "pass_through"
  app_info.pl_device_index = pl.get<unsigned int>("device-index",   app_info.pl_device_index); // default 1
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
	@pkg-config --exists xrt || (echo "xrt is required for the pass_through PL kernel"; exit 1)

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

Before running, finish board setup for your platform: program the required PL + AIE overlay, and make the `pass_through` `.xclbin` available on the board at the path you will reference in `pl-config.xclbin-location` (e.g. `/run/media/mmcblk0p1/x_plus_ml.xclbin`).

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

- **Normal inference** (default HW-tensor mode; forwards outputs through the PL kernel and writes the kernel output):

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json
  ```

- **With INFO logging** — shows the PL device open / xclbin load / kernel-ready lines, useful for first-run debugging:

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --log-level 5
  ```

  Expected PL lines:

  ```
  Opening XRT device 1 for PL kernel 'pass_through'
  Loading xclbin: /run/media/mmcblk0p1/x_plus_ml.xclbin
  PL kernel 'pass_through' ready
  ```

- **Dry run** — validate the config and model without any file I/O or PL forwarding (PL init is skipped):

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --dry-run --log-level 5
  ```

- **Benchmark** for 100 runs — times the full datapath but saves no outputs. It reports the overall average plus a per-stage breakdown (ms/frame): NPU **ML inference**, **data-transfer-to-PL** (host→PL input copy; `0.000` in zero-copy mode), **PL inference** (kernel launch + wait), and **data-transfer-from-PL** (PL output copy→host):

  ```bash
  ml_vart_plus_pl --app-config vart_config_plus_pl.json --benchmark --runs 100
  ```

  Example output:

  ```
  Average inference time over 100 runs: 1.41 ms
  Per-stage average (ms/frame, zero-copy ML->PL):
    ML inference          : 1.410
    data-transfer-to-PL   : 0.000
    PL inference          : 0.328
    data-transfer-from-PL : 0.294
  Run completed successfully.
  ```

- **Inspect model metadata** (no inference) — prints the CPU + HW tensor view and dumps `<model_basename>_info.json`. Use it to look up the input tensor `name`s needed for `ifms-config`:

  ```bash
  ml_vart_plus_pl --get-model-info yolox_nano_onnx_pt_regular_conv_all/yolox_nano_onnx_pt_regular_conv_all.rai
  ```

### 3.6 Verify correctness

Because `pass_through` is an identity kernel, the PL-forwarded outputs must be **byte-for-byte identical** to a plain `ml_vart` run. Run both and compare:

```bash
# Plain inference (no PL) into output/
ml_vart --app-config vart_config.json
# PL-forwarded inference into output_plus_pl/
ml_vart_plus_pl --app-config vart_config_plus_pl.json

# Compare every output tensor
for f in output/*.bin; do
  b=$(basename "$f")
  cmp -s "output/$b" "output_plus_pl/$b" && echo "MATCH: $b" || echo "DIFFER: $b"
done
```

All tensors should report `MATCH`. (For a *real* post-processing kernel the bytes will differ by design; compare against your kernel's expected output instead.)

---

## Command-line arguments

| Option              | Required  | Default | Description                                                  |
| ------------------- | --------- | ------- | ------------------------------------------------------------ |
| `--app-config`      | Mandatory |         | Path to configuration JSON file. Mandatory for inference / dry-run / benchmark flows. Ignored when `--get-model-info` is supplied. |
| `--runs`            | Optional  | `1`     | Number of iterations to run. |
| `--benchmark`       | Optional  | `false` | Benchmark the model for `n` runs. Times the full ML→PL datapath and prints a per-stage breakdown; skips output saving. |
| `--log-level`       | Optional  | `2`     | Log level: `1`=ERROR, `2`=WARNING, `5`=INFO, `6`=DEBUG. |
| `--frames`          | Optional  | all     | Number of frames to process per iteration. |
| `--dry-run`         | Optional  |         | Skip file I/O and PL kernel initialization (test configuration only). |
| `--get-model-info <model-path>` | Optional |  | Standalone mode: print the model's CPU + HW tensor metadata and dump `<model_basename>_info.json`; no inference. `--app-config` is ignored. |
| `--help`            | Optional  |         | Print help and exit. |

---

## Input / output file layout

**Input** — one `.bin` per input tensor, frames concatenated back-to-back in the layout `vart::Runner` expects. Each file size must be an exact multiple of that tensor's per-frame `size_in_bytes` (query with `--get-model-info`). The same frame index across all tensor files forms one batch element; the batch size `N` is fixed by the compiled model. Partial final batches run with fewer frames (no padding).

**Output** — the OFMs written to file are the **PL kernel outputs** (identical bytes for `pass_through`). One `.bin` per output tensor, `N` frames concatenated per inference call. Naming:

- Single run: `infer_out{tensor_idx}-{dtype}_{shape}_{tensor_name}.bin`
- Multiple runs (`--runs > 1`): `iter{run_index}_infer_out{...}.bin` (`{run_index}` 0-based).

---

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `Cannot open library libflexmlrt.so` | `LD_LIBRARY_PATH` missing the flexmlrt lib dir — see [3.3](#33-set-the-runtime-environment). |
| Kernel/xclbin load fails, or `xrt::kernel` throws "no such kernel" | Wrong `device-index` (use `1` on VEK385, see [3.4](#34-pick-the-correct-pl-device-index)), wrong `kernel-name`, or the `.xclbin` does not contain the kernel. |
| `xclbin not found` | `pl-config.xclbin-location` path does not exist on the board. |
| Build error: `load_xclbin` is deprecated | Use the `register_xclbin` + `hw_context` path (see [1.3](#13-write-a-thin-xrt-wrapper-for-the-kernel)). |
| Outputs `DIFFER` from `ml_vart` | Expected only if you replaced `pass_through` with a non-identity kernel; otherwise check byte counts / word rounding / `sync()` calls. |

---

## Related documents

- [json_configs/README.md](json_configs/README.md) — full JSON schema for the app-config (`inference-config`, `ifms-config`, `ofms-dir`, `pl-config`).
- [../ml_vart](../ml_vart) — the plain-inference base example this one extends.
- [../../../reference_design/vek385/rev-b/pass_through](../../../reference_design/vek385/rev-b/pass_through) — the `pass_through` HLS PL kernel source and Vitis build flow.
- [../../docs/runner_options.md](../../../docs/runner_options.md) — the full `runner-options` schema.
