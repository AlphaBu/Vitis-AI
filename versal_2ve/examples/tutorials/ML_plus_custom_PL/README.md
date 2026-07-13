# ML + custom PL inference — end-to-end tutorial (VEK385)

## Copyright and license statement

Copyright (C) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

Note: Example model names, JSON files, IP addresses, part numbers, and commands are for reference only. Modify them for your compiled models, custom PL kernels, and board.

This tutorial describes the **complete flow** for running inference where the **ML part
runs on the NPU** (via the VART‑ML API) and a **custom PL (Programmable Logic) kernel**
runs on the FPGA fabric, with a single host application driving the whole pipeline. It
ties together the reference design (hardware platform, software, Vitis application build)
and an example host application (`ml_vart_plus_pl`) that forwards each NPU output tensor
through a `pass_through` PL kernel.

The datapath is:

```
                 VART-ML (NPU/AIE)                XRT / VART-X (PL)
   IFM ──────►  vart::Runner::run()  ──►  OFM tensor  ──►  custom PL kernel  ──►  result
               (device 0, amdxdna)                       (device 1, PL region)
```

The ML control (VART‑ML) and the PL control (XRT or VART‑X) live **together in the same
host code**, so the application owns the pipeline: it decides when to run the NPU, when to
launch the PL kernel, and how data moves between them.

---

## Prerequisites

- The AMD Vitis / Vivado toolchain (2025.2 or the version matching your reference design).
- The Vitis AI SDK for Versal AI Edge Series Gen 2 (cross toolchain + `vart-ml` / `xrt` `pkg-config` files) for building the host application.
- A VEK385 board (or your target) and the reference design at, e.g.:
  `versal_2ve/reference_design/vek385/rev-b`.

Reference (integrated system reference design build guide):
<https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/integrated_system_reference%20design/build.html>

---

## Flow overview

| Step | Action | Who edits |
|------|--------|-----------|
| 0 | Set up the environment | — |
| 1 | Create the custom **HW platform** (`create_pfm_hw.sh`) | no change if platform unchanged |
| 2 | Create + compile your **PL kernels** against the platform `.xsa` | **user** |
| 3 | Build the **software** with Yocto | no change if SW unchanged |
| 4 | Add your PL kernels to the Vitis app **link** stage (`vitis_prj/Makefile`) | **user** |
| 5 | Write + compile the **host application** (VART‑ML + XRT/VART‑X) | **user** |
| 6 | **Boot** the board with the generated boot images + overlay | — |
| 7 | **Run** the host application on the board | **user** |
| 8 | (Optional) Full‑dataset inference + accuracy | **user** |

---

## 0. Set up the environment

Set up the Vitis / Vivado environment as described in the reference‑design build guide:

<https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/integrated_system_reference%20design/build.html#setting-up-the-environment>

---

## 1. Create the custom HW platform

In the reference‑design directory (for example
`Vitis-AI/versal_2ve/reference_design/vek385/rev-b`), generate the custom hardware
platform:

```bash
cd Vitis-AI/versal_2ve/reference_design/vek385/rev-b
source ./create_pfm_hw.sh
```

**If there is no change to the platform, the user does not need to modify anything** —
just run the script as‑is. This produces the extensible platform used by every later
step:

```
Vitis-AI/versal_2ve/reference_design/vek385/rev-b/hw/example_design_pfm_extensible.xsa
```

Reference:
<https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/integrated_system_reference%20design/build.html#building-vitis-platform>

---

## 2. Create and compile your PL kernels

Create the PL kernel(s) you need for the ML pipeline (e.g. NMS, dequantize, softmax,
resize, letterbox, or any custom post/pre‑processing). Compile each kernel **targeting the
platform generated in Step 1**:

```
Vitis-AI/versal_2ve/reference_design/vek385/rev-b/hw/example_design_pfm_extensible.xsa
```

Each kernel compiles to a `.xo` (e.g. `v++ -c --platform <extensible.xsa> ...`). See the
`pass_through` kernel for a minimal, self‑contained example (source + its own build
Makefile):

- [`../../../reference_design/vek385/rev-b/pass_through`](../../../reference_design/vek385/rev-b/pass_through)

> **Tip — HLS prototyping kernels.** If you are creating a prototyping kernel with Vitis
> HLS, you can use the
> [`vitis-hls-kernel-coding`](../../../skills/vitis-hls-kernel-coding) skill, which covers
> the kernel interface specification (`m_axi` / `s_axilite` / `axis` / block‑level
> control), synthesizable coding restrictions, and optimization (loop pipelining,
> dataflow, memory bursts, array partitioning).

---

## 3. Build the software with Yocto

Build the platform software components with Yocto:

```bash
cd Vitis-AI/versal_2ve/reference_design/vek385/rev-b
source ./create_pfm_sw.sh
```

**If there is no change to the SW components, the user does not need to modify anything.**

Reference:
<https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/integrated_system_reference%20design/build.html#building-software-components-using-yocto>

---

## 4. Add your PL kernels to the Vitis application link stage

Edit the Vitis application build Makefile to include your PL kernel(s) in the **link**
stage so they are linked into the design alongside the AIE graph and the built‑in kernels:

```
Vitis-AI/versal_2ve/reference_design/vek385/rev-b/vitis_prj/Makefile
```

Add each kernel's `.xo` to the `v++` link line. For example (the `pass_through` kernel
added next to `image_processing`):

```make
cd $(ABS_PATH)/link; v++ $(XCXX_COMMON_OPTS) --platform $(PLATFORM) \
    --config $(ABS_PATH)/link/system.cfg \
    -l $(ABS_PATH)/training-libadf.a $(IMAGE_PROCESSING_XO) $(PASS_THROUGH_XO) -o ${PROJECT_NAME}_link.xsa; cd $(ABS_PATH)
```

Make sure your kernel `.xo` is built before the link step (add a build rule / prerequisite
for it), then run the Vitis application build:

```bash
cd Vitis-AI/versal_2ve/reference_design/vek385/rev-b
source ./create_vitis_app.sh
```

> **Note — port connections.**
> If a PL kernel contains **only AXIM (memory) ports**, the tool can connect the AXIM
> memory port to LPDDR **by default** — no manual connection needed.
> If a PL kernel contains **stream (AXIS) ports**, the user must **specify the connections
> themselves** (in the link `system.cfg`, e.g. `stream_connect` / `sc` directives).

---

## 5. Write and compile the host application

Write the host C++ application that drives the pipeline:

- Use the **VART‑ML APIs** (`vart::Runner`) to control ML inference on the NPU.
- Use the **XRT native APIs** *or* the **VART‑X APIs** to control your custom PL kernel.
- Keep both together in the host code: VART‑ML for the ML part, XRT / VART‑X for the PL
  part. **The user controls the pipeline** — the order of NPU runs, PL launches, and the
  data movement between them.

Compile the host code with the installed **SDK** (cross toolchain), producing an AArch64
ELF for the board.

**Example host application** — a complete, documented reference that runs a VART‑ML model
and forwards each output tensor through the `pass_through` PL kernel via the native XRT
C++ API:

- [`../../cpp_examples/ml_vart_plus_pl/README.md`](../../cpp_examples/ml_vart_plus_pl/README.md)

It shows the thin XRT wrapper for a kernel, one‑time kernel init, per‑inference
forwarding, the JSON config (`pl-config`), the Makefile `pkg-config` changes for `xrt`,
and the board run/verify commands. Use it as the starting point for your own host code.

---

## 6. Boot the board

The **boot images** and **overlay** are generated by the previous steps and land in:

```
Vitis-AI/versal_2ve/reference_design/vek385/rev-b/artifact/amd/boot_images
```

Boot and program the board with the generated boot images and overlay. The overlay
provides the combined `.xclbin` (your PL kernels + the AIE/ML overlay); note the on‑board
path you will reference from the host app (e.g. `/run/media/mmcblk0p1/x_plus_ml.xclbin`).

---

## 7. Run the application on the board

Copy the compiled host binary, the model artifact (`.rai`), the input `.bin` file(s), and
the app‑config JSON to the board, set the runtime library path, and run. (Full details in
the [example host README](../../cpp_examples/ml_vart_plus_pl/README.md).)

The example [`ml_vart_plus_pl`](../../cpp_examples/ml_vart_plus_pl) benchmarks the
**yolox_nano_int8** model together with the **`pass_through` PL kernel**. Example command:

```bash
ml_vart_plus_pl --app-config vart_config_plus_pl.json --benchmark --runs 100
```

Example output:

```
Average inference time over 100 runs (ML only): 1.40 ms
Per-stage average (ms/frame, zero-copy ML->PL):
  ML inference             : 1.403
  data-transfer-to-PL      : 0.000
  PL dummy post processing : 0.098
  data-transfer-from-PL    : 0.011
  ------------------------------------
  total (end-to-end)       : 1.513
Run completed successfully.
```

The per‑stage breakdown separates the NPU **ML inference**, the **data‑transfer‑to‑PL**
(host→PL input copy; `0.000` in zero‑copy mode), the **PL dummy post processing** (kernel launch +
wait), and the **data‑transfer‑from‑PL** (PL output copy→host), so you can see exactly
where time goes in the combined ML+PL datapath.

---

## 8. (Optional) Full‑dataset inference + accuracy

To run ML inference **+ PL inference over an entire dataset** (e.g. COCO), the user
prepares the input data as the real system would produce it, and post‑processes the PL
outputs as needed to compute accuracy.

The example provides helper scripts that prepare the ML inference inputs and
post‑process the PL outputs to compute accuracy end‑to‑end (COCO val2017 mAP for
YOLOX‑Nano INT8 over `ml_vart_plus_pl` + `pass_through`):

- [`../../cpp_examples/ml_vart_plus_pl/FULL_PIPELINE.md`](../../cpp_examples/ml_vart_plus_pl/FULL_PIPELINE.md)

---

## Related documents

- [../../cpp_examples/ml_vart_plus_pl/README.md](../../cpp_examples/ml_vart_plus_pl/README.md) — example host application: VART‑ML + XRT PL forwarding, build, run, verify.
- [../../cpp_examples/ml_vart_plus_pl/FULL_PIPELINE.md](../../cpp_examples/ml_vart_plus_pl/FULL_PIPELINE.md) — full COCO val2017 accuracy pipeline (input packing → board run → post‑process → mAP).
- [../../../reference_design/vek385/rev-b](../../../reference_design/vek385/rev-b) — reference design (HW platform, SW, Vitis app build).
- [../../../reference_design/vek385/rev-b/pass_through](../../../reference_design/vek385/rev-b/pass_through) — minimal `pass_through` HLS PL kernel (source + build Makefile).
- [../../../skills/vitis-hls-kernel-coding](../../../skills/vitis-hls-kernel-coding) — skill for writing/optimizing Vitis HLS PL kernels.
- Integrated system reference‑design build guide: <https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/integrated_system_reference%20design/build.html>
