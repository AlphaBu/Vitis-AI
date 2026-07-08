# PL Kernel Test (`pl_test`)

<!--
Copyright (C) 2024-2025 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
-->

A minimal, self-contained test for exercising a **PL (Programmable Logic) kernel**
through the **VART-X** API (`vart::Device`, `vart::Memory`, `vart::PLKernel`), with
**no ML model / VART-ML involved**. It targets the `pass_through` kernel by default,
and the kernel name can be passed on the command line.

`pass_through` is an **identity copy** kernel: it copies `size` 32-bit words from
its input buffer (arg 0) to its output buffer (arg 1). Because the copy is exact,
the test doubles as an easy correctness check — the output must equal the input.

## What the test does

1. Opens the PL device (XRT device index `1` on VEK385) and loads the `.xclbin`.
2. Instantiates the PL kernel by name (default `pass_through`).
3. Prints the kernel's argument layout (`get_config`) — name, type, index, memory bank.
4. Allocates an input and an output buffer on the memory banks the kernel reports.
5. Fills the input with a known ramp: `in[i] = i + 4`.
6. Runs the kernel and waits for completion.
7. Reads back and prints the first 10 output words. For an identity kernel these
   are `4, 5, 6, … 13`.

## Files

| File          | Description                                              |
| ------------- | -------------------------------------------------------- |
| `pl_test.cpp` | The test program (single source file).                  |
| `Makefile`    | Cross-compile rules. Produces the `pl_test` executable.  |

## Prerequisites

- **AMD Vitis AI 6.2 SDK** for the aarch64 (cortexa72/cortexa53) target, providing:
  - `vart-x` (VART-X libraries — `pkg-config vart-x`)
  - `xrt` (`pkg-config xrt`)
- A VEK385 board running the matching image, with an `.xclbin` on the board that
  contains the target PL kernel (e.g. `pass_through`).

## Building

Cross-compile with the SDK environment sourced:

```sh
unset LD_LIBRARY_PATH   # avoid host libs leaking into the cross build
source /proj/xcohdstaff3/brucey/nobkup/VAIML/vai_6.2_sdk/environment-setup-cortexa72-cortexa53-amd-linux

make clean
make all
```

This produces the `pl_test` executable (ELF aarch64). The `Makefile` only depends
on `vart-x` and `xrt` via `pkg-config`; both headers and libraries come from the
SDK sysroot, so no hardcoded include paths are needed.

To (optionally) install into the SDK sysroot's `/usr/bin`:

```sh
make install
```

## Running on the board

The build tree lives under `/proj/...`, which is **not** NFS-mounted on the board,
so copy the binary over (the `.xclbin` is already on the board):

```sh
# from the build host
scp -o StrictHostKeyChecking=no -o PubkeyAuthentication=no \
    pl_test root@10.25.38.208:/tmp/pl_test
```

Then on the board (or via `ssh root@10.25.38.208`):

```sh
# VART-X needs libflexmlrt.so on the library path
export LD_LIBRARY_PATH=/usr/lib/python3.12/site-packages/flexmlrt/lib:$LD_LIBRARY_PATH

# usage: pl_test <xclbin-location> [kernel-name]
/tmp/pl_test /run/media/mmcblk0p1/x_plus_ml.xclbin pass_through

# the kernel name is optional; it defaults to "pass_through":
/tmp/pl_test /run/media/mmcblk0p1/x_plus_ml.xclbin
```

### Command-line arguments

| Position | Argument          | Required | Default        | Description                                        |
| -------- | ----------------- | -------- | -------------- | -------------------------------------------------- |
| 1        | `xclbin-location` | Yes      | —              | Path to the `.xclbin` on the board.                |
| 2        | `kernel-name`     | No       | `pass_through` | Name of the PL kernel to instantiate from the xclbin. |

The PL device index is fixed at `1` (the PL region on VEK385; the NPU is device 0).
The transfer size is `602112` bytes (`TRANSFER_SIZE_BYTES` in `pl_test.cpp`); change
that constant to test a different size (must be a multiple of 4).

### Expected output

```
xclbin      : /run/media/mmcblk0p1/x_plus_ml.xclbin
kernel-name : pass_through
device-index: 1
Argument Name: in_r
Argument type: non scalar
Argument data_type: void*
Argument Index: 0
Argument size: 8
Memory Index: 3
==============
Argument Name: out_r
Argument type: non scalar
Argument data_type: void*
Argument Index: 1
Argument size: 8
Memory Index: 3
==============
Argument Name: size
Argument type: scalar
Argument data_type: unsigned int
Argument Index: 2
Argument size: 0
Memory Index: -1
==============
out[0] = 4
out[1] = 5
...
out[9] = 13
```

`out[i] == i + 4` confirms the full **host → `vart::Memory` → PL kernel → `vart::Memory` → host**
data path is working and that `pass_through` performs an exact identity copy.

## How it works (VART-X API walkthrough)

```cpp
// 1. Open the PL device and load the xclbin (cached per device+xclbin).
auto device = vart::Device::get_device_hdl(DEFAULT_DEVICE_INDEX, xclbin_location);

// 2. Instantiate the kernel by name.
std::string json_data = "{}";
auto* plkernel = new vart::PLKernel(
    vart::PLKernelImplType::PL_KERNEL_XRT, kernel_name, json_data, device);

// 3. Discover the kernel's arguments (and which memory bank each is on).
std::vector<vart::ArgumentInfo> arg_info_list;
plkernel->get_config(arg_info_list);

// 4. Allocate buffers on the reported banks.
auto in_mem  = std::make_shared<vart::Memory>(
    vart::MemoryImplType::XRT,
    static_cast<size_t>(TRANSFER_SIZE_BYTES),
    static_cast<uint8_t>(arg_info_list[0].mem_index), device);
auto out_mem = std::make_shared<vart::Memory>(
    vart::MemoryImplType::XRT,
    static_cast<size_t>(TRANSFER_SIZE_BYTES),
    static_cast<uint8_t>(arg_info_list[1].mem_index), device);

// 5. Map, write the input ramp, unmap.
auto* p = (uint32_t*)in_mem->map(vart::DataMapFlags::WRITE);
for (...) p[i] = i + 4;
in_mem->unmap();

// 6. Launch with the buffers' physical addresses + the word count, then wait.
plkernel->process((void*)in_mem->get_physical_addr(),
                  (void*)out_mem->get_physical_addr(),
                  TRANSFER_SIZE_BYTES / 4);   // pass_through "size" is a WORD count
plkernel->wait(1000);

// 7. Map the output for reading.
auto* q = (uint32_t*)out_mem->map(vart::DataMapFlags::READ);
```

### Important: `vart::Memory` constructor overload

`vart::Memory` has two 4-argument constructors:

```cpp
Memory(MemoryImplType type, size_t size,  uint8_t mbank_idx, shared_ptr<Device>); // allocation
Memory(MemoryImplType type, int dma_fd,   size_t  size,      shared_ptr<Device>); // dma-buf import
```

If you pass the size and bank as plain `int`, overload resolution binds to the
**dma-buf import** constructor (`int → int dma_fd` is an exact match), so the size
is mistaken for a file descriptor and you get:

```
Memory instance creation from FD failed: vvas_memory_alloc_from_fd() failed - invalid FD or not CMA-backed
```

Cast the size to `size_t` and the bank to `uint8_t` (as shown above) so the
**allocation** constructor is selected.

## Troubleshooting

| Symptom                                                                        | Cause / Fix                                                                                                   |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `Cannot open library libflexmlrt.so`                                           | `export LD_LIBRARY_PATH=/usr/lib/python3.12/site-packages/flexmlrt/lib:$LD_LIBRARY_PATH` before running.       |
| `vvas_memory_alloc_from_fd() failed - invalid FD or not CMA-backed`            | The `vart::Memory` size/bank args were `int`; cast to `size_t`/`uint8_t` (see above).                          |
| Kernel not found / instantiation throws                                        | The `kernel-name` isn't in the `.xclbin`. Check the argument dump / the xclbin contents.                       |
| `No such file or directory` for the xclbin                                     | Wrong `xclbin-location`; confirm the path on the board (e.g. `/run/media/mmcblk0p1/x_plus_ml.xclbin`).         |
| Output does not match the input ramp                                           | The named kernel is not an identity copy, or the transfer size / word count is wrong.                          |
