/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software
 * is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
 * KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL "AMD" BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
 * OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. Except as contained in this notice, the name of the AMD shall
 * not be used in advertising or otherwise to promote the sale, use or other
 * dealings in this Software without prior written authorization from AMD.
 */

/*
 * Standalone PL-kernel test for the "pass_through" kernel.
 *
 * The pass_through kernel is an identity copy kernel: it copies "size" 32-bit
 * words from its input buffer (arg 0) to its output buffer (arg 1). This test
 * allocates an input/output buffer pair on the memory banks reported by the
 * kernel, fills the input with a known ramp, runs the kernel, and prints back
 * the first few output words. For an identity kernel out[i] == in[i].
 *
 * Usage:
 *   pl_test <xclbin-location> [kernel-name]
 *     xclbin-location : path to the .xclbin that contains the PL kernel
 *     kernel-name     : PL kernel to run (default: "pass_through")
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <vart/vart_device.hpp>
#include <vart/vart_memory.hpp>
#include <vart/vart_memory_types.hpp>
#include <vart/vart_plkernel.hpp>
#include <vart/vart_plkernel_types.hpp>

using namespace std;

// On VEK385 the PL region is XRT device index 1 (the NPU is device 0).
#define DEFAULT_DEVICE_INDEX 1

// Default PL kernel name; can be overridden on the command line.
#define DEFAULT_KERNEL_NAME "pass_through"

// Transfer size in bytes (must be a multiple of 4 for a 32-bit word kernel).
#define TRANSFER_SIZE_BYTES 602112

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <xclbin-location> [kernel-name]"
              << std::endl;
    std::cerr << "  xclbin-location : path to the .xclbin containing the kernel"
              << std::endl;
    std::cerr << "  kernel-name     : PL kernel to run (default: \""
              << DEFAULT_KERNEL_NAME << "\")" << std::endl;
    return 1;
  }

  std::string xclbin_location = argv[1];
  std::string kernel_name = (argc >= 3) ? argv[2] : DEFAULT_KERNEL_NAME;

  std::cout << "xclbin      : " << xclbin_location << std::endl;
  std::cout << "kernel-name : " << kernel_name << std::endl;
  std::cout << "device-index: " << DEFAULT_DEVICE_INDEX << std::endl;

  vector<shared_ptr<vart::Memory>> pl_in_memory_vec;
  vector<shared_ptr<vart::Memory>> pl_out_memory_vec;

  auto device =
      vart::Device::get_device_hdl(DEFAULT_DEVICE_INDEX, xclbin_location);

  string json_data = "{}";
  vart::PLKernel* plkernel = new vart::PLKernel(
      vart::PLKernelImplType::PL_KERNEL_XRT, kernel_name, json_data, device);

  vector<vart::ArgumentInfo> arg_info_list;
  plkernel->get_config(arg_info_list);

  for (const auto& arg_info : arg_info_list) {
    std::cout << "Argument Name: " << arg_info.arg_name << std::endl;
    std::cout << "Argument type: " << arg_info.arg_type << std::endl;
    std::cout << "Argument data_type: " << arg_info.arg_data_type << std::endl;
    std::cout << "Argument Index: " << arg_info.arg_index << std::endl;
    std::cout << "Argument size: " << arg_info.arg_size << std::endl;
    std::cout << "Memory Index: " << arg_info.mem_index << std::endl;
    std::cout << "============== " << std::endl;
  }

  // pass_through has two non-scalar (buffer) arguments: input at arg 0 and
  // output at arg 1. Allocate a buffer on each argument's memory bank.
  //
  // NOTE: the size/bank arguments are cast to size_t/uint8_t so that overload
  // resolution selects the allocation constructor
  //   Memory(MemoryImplType, size_t size, uint8_t mbank_idx, device)
  // and NOT the dma-buf import constructor
  //   Memory(MemoryImplType, int dma_fd, size_t size, device),
  // which an unqualified int size would bind to (treating the size as an FD).
  pl_in_memory_vec.push_back(make_shared<vart::Memory>(
      vart::MemoryImplType::XRT, static_cast<size_t>(TRANSFER_SIZE_BYTES),
      static_cast<uint8_t>(arg_info_list[0].mem_index) /* input arg 0 */,
      device));
  pl_out_memory_vec.push_back(make_shared<vart::Memory>(
      vart::MemoryImplType::XRT, static_cast<size_t>(TRANSFER_SIZE_BYTES),
      static_cast<uint8_t>(arg_info_list[1].mem_index) /* output arg 1 */,
      device));

  // Fill the input buffer with a known ramp: in[i] = i + 4.
  uint32_t* pl_in_memory = const_cast<uint32_t*>(
      reinterpret_cast<const uint32_t*>(
          pl_in_memory_vec[0]->map(vart::DataMapFlags::WRITE)));
  for (int i = 0; i < TRANSFER_SIZE_BYTES / 4; i++) {
    pl_in_memory[i] = i + 4;
  }
  pl_in_memory_vec[0]->unmap();

  void* mem_in = (void*)(uintptr_t)(pl_in_memory_vec[0]->get_physical_addr());
  void* mem_out = (void*)(uintptr_t)(pl_out_memory_vec[0]->get_physical_addr());
  unsigned int size = TRANSFER_SIZE_BYTES / 4;  // pass_through "size" is a word count.

  plkernel->process(mem_in, mem_out, size);
  plkernel->wait(1000);

  // Read back and print the first few output words. For an identity kernel the
  // expected values are 4, 5, 6, ... (matching the input ramp).
  uint32_t* mapped_memory_out = const_cast<uint32_t*>(
      reinterpret_cast<const uint32_t*>(
          pl_out_memory_vec[0]->map(vart::DataMapFlags::READ)));
  for (int i = 0; i < 10; i++) {
    std::cout << "out[" << i << "] = " << mapped_memory_out[i] << std::endl;
  }
  pl_out_memory_vec[0]->unmap();

  delete plkernel;
  return 0;
}
