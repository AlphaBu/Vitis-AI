# Inference Configuration JSON Guide

<!--
## Copyright and license statement

Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
-->


This document explains the structure and usage of the `ml_vart_config.json` file for configuring model inference runs.

## Overview

The JSON file contains an object that describes the configuration for a single model inference session.

Field requirements depend on CLI mode:

- Always required: `inference-config.model-file`
- Conditionally required: `ifms-config` (required when `--dry-run` is **not** set)
- Conditionally required: `pl-config.xclbin-location` (required when `--dry-run` is **not** set; identifies the `.xclbin` that contains the `pass_through` PL kernel used to post-process the inference outputs)
- Optional: `ofms-dir` (defaults to `"output"` if not specified; only used during normal inference, ignored in `--dry-run` and `--benchmark` modes)

Let us consider the following example configuration for a ResNet50 model, which accepts one input feature map (IFM).

## JSON Structure Example

```json
{
  "inference-config": {
    "model-file": "/etc/vai/models/resnet50_int8/resnet50_int8.rai",
    "runner-options": {
      "log-level": "WARNING",
      "ai-analyzer-profiling": false
    }
  },
  "ifms-config": [
    {
      "name": "input",
      "file": "/etc/vai/models/resnet50_int8/data/ifm_input_int8_1x224x224x4.bin"
    }
  ],
  "ofms-dir": "output",
  "pl-config": {
    "xclbin-location": "/etc/vai/ml_vart_plus_pl/x_plus_ml.xclbin",
    "kernel-name": "pass_through",
    "device-index": 1
  }
}
```

### Description of JSON Fields

#### Description of `inference-config` Object

| Field            | Type   | Description                                                            | Example Value                                       |
| ---------------- | ------ | ---------------------------------------------------------------------- | --------------------------------------------------- |
| `model-file`     | String | Path to the compiled model artifact (`.rai`) or model cache directory. | `"/etc/vai/models/resnet50_int8/resnet50_int8.rai"` |
| `runner-options` | Object | All `vart::Runner` specific options.                                   | See below                                           |

See [runner_options.md](../../../docs/runner_options.md) for the full `runner-options` schema (fields, defaults, NPU column placement, and auto-placement policy).

#### Description of `ifms-config` Array

`ifms-config` is an array containing one entry per input tensor of the model.

Entries are bound to the `vart::Runner` input tensors **by name** (not by JSON-array order). The `name` field of each entry must match a `vart::Runner`-reported input tensor name; the array may be authored in any order. The application aborts at startup if any `name` does not match a `vart::Runner` input tensor, if a `name` is duplicated, or if the entry count differs from the model's input-tensor count. Use `ml_vart_plus_pl --get-model-info <model-path>` to inspect the `vart::Runner`-reported input tensor names.

Requirement:

- Required when `--dry-run` is not set.
- Optional when `--dry-run` is set.

| Field  | Type   | Description                                | Example Value                                                         |
| ------ | ------ | ------------------------------------------ | --------------------------------------------------------------------- |
| `name` | String | `vart::Runner`-reported input tensor name this entry binds to. Must match exactly. | `"input"`                                                             |
| `file` | String | Path to the IFM `.bin` file for that tensor. File must exist and have a `.bin` extension. | `"/etc/vai/models/resnet50_int8/data/ifm_input_int8_1x224x224x4.bin"` |

#### Description of `ofms-dir` Field

| Field      | Type   | Description                                                                                                                                                                                                                                                                              | Example Value |
| ---------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `ofms-dir` | String | Path to the directory where OFMs will be dumped. If not specified, defaults to `"output"`. Output directory is created automatically if it does not exist. **Note:** This field is only used during normal inference; it is ignored when `--dry-run` or `--benchmark` modes are enabled. | `"output"`    |

#### Description of `pl-config` Object

`pl-config` configures the `pass_through` PL (Programmable Logic) kernel that the inference outputs are forwarded through. After `vart::Runner` produces the output tensors, each output tensor's bytes are transferred to the kernel, the kernel is launched, and the kernel result is written back over the tensor buffer so that the saved OFMs are the PL kernel output.

Requirement:

- `xclbin-location` is required when `--dry-run` is not set (the PL kernel is not initialized in `--dry-run` mode).

| Field             | Type   | Description                                                                                                 | Example Value                                  |
| ----------------- | ------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `xclbin-location` | String | Path to the `.xclbin` on the board that contains the `pass_through` PL kernel. File must exist at runtime. | `"/etc/vai/ml_vart_plus_pl/x_plus_ml.xclbin"`  |
| `kernel-name`     | String | Name of the PL kernel to instantiate from the `.xclbin`. Defaults to `"pass_through"` if not specified.    | `"pass_through"`                               |
| `device-index`    | Integer | XRT device index that hosts the PL kernel. Defaults to `1` if not specified (on VEK385 the PL region is device `1` and the NPU is device `0`). | `1`                                            |
