/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * EVENT SHALL AMD BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
 * OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. Except as contained in this notice, the name of the AMD shall
 * not be used in advertising or otherwise to promote the sale, use or other
 * dealings in this Software without prior written authorization from AMD.
 */

#include <getopt.h>
#include <array>
#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <vart/vart_runner_factory.hpp>
#include "common/app_logger.hpp"

/* Native XRT API used to drive the pass_through PL (Programmable Logic)
 * kernel that post-processes the VART inference outputs. */
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>
#include <xrt/experimental/xrt_xclbin.h>

using namespace std;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using boost::property_tree::ptree;

/**********************************************************************************
 * ml_vart - VART-ML Reference Application
 **********************************************************************************
 * This application demonstrates VART-based inference using a VAIML compiled model.
 * It loads input tensors from binary files, executes inference via vart::Runner,
 * and saves output tensors. Configuration is specified via a mandatory JSON file.
 *
 * Key Components:
 * - app_opt:          Application options (CLI + app I/O settings)
 * - runner_opt:       Runner-specific options loaded from JSON
 * - app_context: Main context managing runner, tensors, and inference loop
 * - utils:            Static helpers for file I/O operations
 * - vart::Runner:     VART API interface for model execution
 *
 * Execution Modes:
 * - Normal:    Full inference with file I/O
 * - Dry-run:   Skip file I/O (for testing configuration)
 * - Benchmark: Measure performance without saving outputs
 *
 * Flow:
 * 1. Parse command-line args and JSON config
 * 2. Create VART runner and allocate tensors
 * 3. For each iteration:
 *    - Load input data from IFM files
 *    - Execute inference (batched, handles partial batches)
 *    - Save output data to OFM files
 * 4. Report performance metrics (if benchmark mode)
 **********************************************************************************/

namespace {
static void print_help_text(char* pn) {
  std::cout << "Usage: " << pn << " --app-config <file> [OPTIONS]" << std::endl;
  std::cout << "       " << pn << " --get-model-info <model-path> [OPTIONS]\n" << std::endl;
  std::cout << "OPTIONS:" << std::endl;
  std::cout << "  --app-config <file>\tJSON configuration file (mandatory for inference;" << std::endl;
  std::cout << "                       \tignored when --get-model-info <model-path> is supplied)" << std::endl;
  std::cout << "  --runs <N>\t\tNumber of iterations (default: 1)" << std::endl;
  std::cout << "  --frames <N>\t\tFrames to process per iteration (default: -1, all)" << std::endl;
  std::cout << "  --benchmark\t\tMeasure performance without saving outputs" << std::endl;
  std::cout << "  --dry-run\t\tSkip file I/O (test configuration)" << std::endl;
  std::cout << "  --log-level <N>\tLog level: 1=ERROR, 2=WARNING, 5=INFO, 6=DEBUG (default: 2)" << std::endl;
  std::cout << "  --get-model-info <model-path>" << std::endl;
  std::cout << "                       \tStandalone mode: create the runner for <model-path>, query" << std::endl;
  std::cout << "                       \tthe model's tensor metadata (both HW and CPU views), print" << std::endl;
  std::cout << "                       \tit to the console and dump the same content to" << std::endl;
  std::cout << "                       \t<model_basename>_info.json in the current directory." << std::endl;
  std::cout << "                       \t<model-path> is either a .rai file or the compiled-" << std::endl;
  std::cout << "                       \tmodel root directory containing vaiml_par_0. No inference" << std::endl;
  std::cout << "                       \tis performed. --app-config is ignored if also supplied." << std::endl;
  std::cout << "                       \tThe runner is always created with CPU input/output tensor" << std::endl;
  std::cout << "                       \ttype." << std::endl;
  std::cout << "  --help\t\tShow this help\n" << std::endl;
  std::cout << "EXAMPLES:" << std::endl;
  std::cout << "  " << pn << " --app-config config.json" << std::endl;
  std::cout << "  " << pn << " --app-config config.json --benchmark --runs 100" << std::endl;
  std::cout << "  " << pn << " --app-config config.json --frames 50 --log-level 6" << std::endl;
  std::cout << "  " << pn << " --get-model-info /etc/vai/models/resnet50_int8/resnet50_int8.rai" << std::endl;
}
}  // namespace

/**
 * @enum vart_app_status
 * @brief provides the status of the vart application.
 * - SUCCESS:   Indicates operation is successful.
 * - FAILURE:   Indicates operation has failed.
 */
enum class vart_app_status : int { SUCCESS = 0, FAILURE = -1 };

/**
 * @struct runner_opt
 * @brief Structure to hold runner-specific options loaded from JSON.
 *
 * This structure encapsulates all runner configuration parameters required
 * to initialize and run vart::Runner.
 *
 * Members:
 * - model_cache_dir: Path to the directory where the model cache is stored.
 * - log_level: Sets the runner's log level for message output.
 *   Accepted values: "ERROR", "WARNING", "INFO", "DEBUG". Default "INFO".
 * - config_json: Path to the Vitis AI configuration file.
 * - aie_columns_sharing: Specify how to schedule column resources while loading
 *   the model. true indicates shared mode, false indicates exclusive mode. Default true.
 * - start_column: Starting column for NPU overlay execution. By default loads
 *   model on the available column.
 * - is_start_column_set: Whether start_column was explicitly configured.
 * - cma_index: CMA index on which buffer objects should be allocated by the
 *   current runner. Default is CMA index 0.
 * - is_cma_index_set: Whether cma_index was explicitly configured.
 * - ai_analyzer_profiling: Enable or disable AI Analyzer profiling. Default false.
 */
struct runner_opt {
  std::string model_cache_dir;
  std::string log_level;
  std::string config_json;
  bool aie_columns_sharing;
  uint32_t start_column;
  bool is_start_column_set;
  int32_t cma_index;
  bool is_cma_index_set;
  bool ai_analyzer_profiling;
  std::string input_tensor_type;
  std::string output_tensor_type;

  /**
   * @brief Constructor to initialize model configuration with default values
   */
  runner_opt() {
    model_cache_dir = "";
    log_level = "WARNING";
    config_json = "";
    aie_columns_sharing = true;
    start_column = 0;
    is_start_column_set = false;
    cma_index = 0;
    is_cma_index_set = false;
    ai_analyzer_profiling = false;
    input_tensor_type = "HW";
    output_tensor_type = "HW";
  }

}; /* runner_opt */

/**
 * @struct app_opt
 * @brief Structure to hold application-specific options.
 */
struct app_opt {
  /* Map keyed by ifms-config[i].name (which must match the runner's
   * input tensor name) -> ifms-config[i].file. Populated at JSON parse
   * time; consumed once by app_context::resolve_ifms_in_runner_order()
   * after the runner is up, then no longer used. */
  std::unordered_map<std::string, std::string> ifm_files_by_name;
  std::filesystem::path ofm_dir;
  bool dry_run;
  std::string app_config_file;
  uint32_t n_runs;
  int frames;
  AppLogLevel app_log;
  /* --get-model-info: when true the application creates the VART runner,
   * queries the runner-reported tensor metadata for both HW and CPU
   * tensor types (independent of the user's runner-options choice),
   * prints a summary to the console, dumps `<model_basename>_info.json`
   * in the current working directory and exits before allocating any
   * IFM/OFM tensors. */
  bool get_model_info;
  /* Standalone model path supplied directly via --get-model-info <path>.
   * Used only when --app-config is not provided; lets the operator inspect
   * a model without authoring an app-config JSON. */
  std::string get_model_info_path;

  /* pass_through PL kernel configuration (from JSON "pl-config").
   * - pl_xclbin: path to the .xclbin that contains the PL kernel. Mandatory
   *   for the inference flow because every inference output is forwarded
   *   through the kernel and the kernel output becomes the app output.
   * - pl_kernel: kernel top-function name (default "pass_through").
   * - pl_device_index: XRT device index that hosts the PL kernel
   *   (default 1; on VEK385 the PL region is device 1, the NPU is 0). */
  std::string pl_xclbin;
  std::string pl_kernel;
  unsigned int pl_device_index;

  app_opt() {
    ifm_files_by_name.clear();
    ofm_dir = "";
    app_config_file = "";
    dry_run = false;
    n_runs = 1;
    frames = -1;
    app_log = AppLogLevel::WARNING;
    get_model_info = false;
    get_model_info_path = "";
    pl_xclbin = "";
    pl_kernel = "pass_through";
    pl_device_index = 1;
  }

  /**
   * @brief Validate app and model paths after JSON parsing.
   * @return Status of the validation.
   */
  vart_app_status validate_app_opt(const runner_opt& runner) {
    // List of paths to check: pair of (path, error message)
    std::vector<std::pair<std::string, std::string>> paths = {{runner.model_cache_dir, "Model cache dir not found:"},
                                                              {ofm_dir.string(), "OFM directory not found, creating:"}};
    try {
      /* validate IFMs: file existence + .bin extension. This runs at
       * parse time, before the runner is created, so we cannot
       * cross-check the JSON 'name' values against the runner-reported
       * input tensor names here. That cross-check happens later in
       * app_context::resolve_ifms_in_runner_order(); do not duplicate
       * it here. */
      for (const auto& kv : ifm_files_by_name) {
        const std::string& ifm = kv.second;
        if (!fs::exists(ifm)) {
          APP_LOG(AppLogLevel::ERROR, app_log, "IFM file not found: %s (ifms-config name '%s')", ifm.c_str(),
                  kv.first.c_str());
          return vart_app_status::FAILURE;
        }

        /* check if the extension is .bin */
        if (fs::path(ifm).extension() != ".bin") {
          APP_LOG(AppLogLevel::ERROR, app_log, "IFM file must have .bin extension: %s (ifms-config name '%s')",
                  ifm.c_str(), kv.first.c_str());
          return vart_app_status::FAILURE;
        }
      }

      for (size_t i = 0; i < paths.size(); ++i) {
        const auto& path = paths[i].first;
        const auto& msg = paths[i].second;

        if (!fs::exists(path)) {
          // Only create OFM dir if missing, others return failure
          if (i == 1) {
            fs::create_directories(path);
            APP_LOG(AppLogLevel::WARNING, app_log, "OFM Dir path not present, Created OFM directory: %s", path.c_str());
          } else {
            APP_LOG(AppLogLevel::ERROR, app_log, "%s %s", msg.c_str(), path.c_str());
            return vart_app_status::FAILURE;
          }
        }
      }
    } catch (...) {
      APP_LOG(AppLogLevel::ERROR, app_log, "Exception occurred while validating paths.");
      return vart_app_status::FAILURE;
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Print application options
   */
  void print() {
    for (const auto& kv : ifm_files_by_name) {
      APP_LOG(AppLogLevel::INFO, app_log, "ifm_file [name='%s']: %s", kv.first.c_str(), kv.second.c_str());
    }
    APP_LOG(AppLogLevel::INFO, app_log, "ofm_dir: %s", ofm_dir.string().c_str());
    APP_LOG(AppLogLevel::INFO, app_log, "dry_run: %s", dry_run ? "true" : "false");
    APP_LOG(AppLogLevel::INFO, app_log, "pl_xclbin: %s", pl_xclbin.c_str());
    APP_LOG(AppLogLevel::INFO, app_log, "pl_kernel: %s", pl_kernel.c_str());
    APP_LOG(AppLogLevel::INFO, app_log, "pl_device_index: %u", pl_device_index);
    APP_LOG(AppLogLevel::INFO, app_log, "******************************************");
  }

}; /* app_opt */

/**
 * @brief Print runner options
 */
static void print_runner_opt(const runner_opt& opt, AppLogLevel app_log) {
  APP_LOG(AppLogLevel::INFO, app_log, "model_cache_dir: %s", opt.model_cache_dir.c_str());
  APP_LOG(AppLogLevel::INFO, app_log, "log_level: %s", opt.log_level.c_str());
  APP_LOG(AppLogLevel::INFO, app_log, "config_json: %s", opt.config_json.c_str());
  APP_LOG(AppLogLevel::INFO, app_log, "aie_columns_sharing: %s", opt.aie_columns_sharing ? "true" : "false");
  if (opt.is_start_column_set)
    APP_LOG(AppLogLevel::INFO, app_log, "start_column: %u", opt.start_column);
  if (opt.is_cma_index_set)
    APP_LOG(AppLogLevel::INFO, app_log, "cma_index: %d", opt.cma_index);
  APP_LOG(AppLogLevel::INFO, app_log, "ai_analyzer_profiling: %s", opt.ai_analyzer_profiling ? "true" : "false");
  APP_LOG(AppLogLevel::INFO, app_log, "input_tensor_type: %s", opt.input_tensor_type.c_str());
  APP_LOG(AppLogLevel::INFO, app_log, "output_tensor_type: %s", opt.output_tensor_type.c_str());
  APP_LOG(AppLogLevel::INFO, app_log, "******************************************");
}

/**
 * @class utils
 * @brief Utility class for file operations
 */
class utils {
 public:
  /**
   * @brief Read a string array from the property tree
   *
   * @note This utility function is available for parsing JSON arrays but is not currently
   *       used in the main configuration parsing flow. The parse_config_file() function
   *       directly iterates over ptree children instead. This function is retained for
   *       API completeness and potential future use.
   *
   * @param tree The property tree
   * @param key The key to read the array from
   * @return A vector of strings containing the array elements
   */
  static std::vector<std::string> read_string_array(const boost::property_tree::ptree& tree, const std::string& key) {
    std::vector<std::string> out;
    auto child_opt = tree.get_child_optional(key);
    if (!child_opt)
      return out;
    for (const auto& kv : child_opt.get()) {
      // Each array element in ptree is stored under empty key with a child that has data()
      out.emplace_back(kv.second.get_value<std::string>(""));
    }
    return out;
  }

  /**
   * @brief Load binary data from a file into a buffer.
   *
   * This function reads binary data from the specified file and loads it into the provided buffer.
   * It supports reading data from a specific offset within the file.
   *
   * @param filename Path to the binary file to be read.
   * @param data Pointer to the buffer where the binary data will be loaded.
   *             The buffer must be pre-allocated and large enough to hold the specified size of data.
   * @param size The number of bytes to read from the file.
   * @param offset The offset (in bytes) from the beginning of the file where reading starts.
   *               Defaults to 0, meaning reading starts from the beginning of the file.
   * @return true if the data is successfully loaded into the buffer, false otherwise.
   *         Returns false in case of file read errors, invalid file paths, or insufficient data in the file.
   */
  static bool load_binary_data(const std::filesystem::path& filename, void* data, size_t size, size_t offset = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      // Move the file pointer to the specified offset
      file.seekg(offset, std::ios::beg);
      if (!file) {
        APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to seek to offset %zu in file %s", offset,
                filename.c_str());
        file.close();
        return false;
      }

      // Read the data from the file
      file.read(static_cast<char*>(data), size);
      if (!file) {
        APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to read %zu bytes from file %s at offset %zu", size,
                filename.c_str(), offset);
        file.close();
        return false;
      }

      file.close();
    } else {
      APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to open file %s", filename.c_str());
      return false;
    }
    return true;
  }

  /**
   * @brief Save binary data from a buffer to a file.
   *
   * This function writes binary data from the provided buffer to the specified file.
   * It supports writing data at a specific offset within the file. If the file does not exist,
   * it will be created. If the file already exists, the data will be written starting at the
   * specified offset, potentially overwriting existing content.
   *
   * @note This utility function is available for general use but is not currently used in the
   *       main inference output flow. Output files are now managed via open_output_files(),
   *       write_output_tensors(), and close_output_files() which keep files open during
   *       iteration for better performance. This function is retained for API completeness
   *       and symmetry with load_binary_data().
   *
   * @param filename Path to the binary file where the data will be saved.
   * @param data Pointer to the buffer containing the binary data to be saved.
   *             The buffer must contain at least `size` bytes of valid data.
   * @param size The number of bytes to write to the file.
   * @param offset The offset (in bytes) from the beginning of the file where writing starts.
   *               Defaults to 0, meaning writing starts from the beginning of the file.
   * @param truncate If true, truncates the file before writing. Defaults to false.
   * @return true if the data was successfully written, false otherwise.
   *
   * @note Ensure that the file path is valid and that the application has write permissions
   *       for the specified file.
   */
  static bool save_binary_data(const std::filesystem::path& filename,
                               const void* data,
                               size_t size,
                               size_t offset = 0,
                               bool truncate = false) {
    std::fstream file;

    // Determine the appropriate file open mode
    std::ios_base::openmode mode = std::ios::binary;

    if (truncate) {
      // Truncate mode: create new file or overwrite existing
      mode |= std::ios::out | std::ios::trunc;
    } else if (std::filesystem::exists(filename)) {
      // Existing file: open for reading and writing to allow seeking
      mode |= std::ios::in | std::ios::out;
    } else {
      // New file: create it
      mode |= std::ios::out;
    }

    file.open(filename, mode);

    if (!file.is_open()) {
      APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to open or create file: %s", filename.c_str());
      return false;
    }

    // Move the file pointer to the specified offset
    file.seekp(offset, std::ios::beg);
    if (!file) {
      APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to seek to offset %zu in file %s", offset,
              filename.c_str());
      file.close();
      return false;
    }

    // Write the data to the file
    file.write(static_cast<const char*>(data), size);
    if (!file) {
      APP_LOG(AppLogLevel::ERROR, AppLogLevel::ERROR, "Unable to write data to file %s at offset %zu", filename.c_str(),
              offset);
      file.close();
      return false;
    }

    file.close();
    return true;
  }
};

/*
 * Convert vart::DataType to a short lowercase string (e.g. "int8", "fp32").
 */
static std::string get_data_type_string(vart::DataType data_type) {
  switch (data_type) {
    case vart::DataType::BOOLEAN:
      return "boolean";
    case vart::DataType::INT8:
      return "int8";
    case vart::DataType::UINT8:
      return "uint8";
    case vart::DataType::INT16:
      return "int16";
    case vart::DataType::UINT16:
      return "uint16";
    case vart::DataType::BF16:
      return "bf16";
    case vart::DataType::FP16:
      return "fp16";
    case vart::DataType::INT32:
      return "int32";
    case vart::DataType::UINT32:
      return "uint32";
    case vart::DataType::FLOAT32:
      return "fp32";
    case vart::DataType::INT64:
      return "int64";
    case vart::DataType::UINT64:
      return "uint64";
    default:
      return "UNKNOWN";
  }
}

/*
 * Convert vart::MemoryLayout to its symbolic string (e.g. "NCHW", "HCWNC4").
 */
static std::string get_memory_layout_string(vart::MemoryLayout layout) {
  switch (layout) {
    case vart::MemoryLayout::NC:
      return "NC";
    case vart::MemoryLayout::NCH:
      return "NCH";
    case vart::MemoryLayout::NHC:
      return "NHC";
    case vart::MemoryLayout::NHW:
      return "NHW";
    case vart::MemoryLayout::NHWC:
      return "NHWC";
    case vart::MemoryLayout::NCHW:
      return "NCHW";
    case vart::MemoryLayout::NHWC4:
      return "NHWC4";
    case vart::MemoryLayout::NHWC8:
      return "NHWC8";
    case vart::MemoryLayout::NC4HW4:
      return "NC4HW4";
    case vart::MemoryLayout::NC8HW8:
      return "NC8HW8";
    case vart::MemoryLayout::HCWNC4:
      return "HCWNC4";
    case vart::MemoryLayout::HCWNC8:
      return "HCWNC8";
    case vart::MemoryLayout::HCWNC16:
      return "HCWNC16";
    case vart::MemoryLayout::NHW16C4WC:
      return "NHW16C4WC";
    case vart::MemoryLayout::GENERIC:
      return "GENERIC";
    default:
      return "UNKNOWN";
  }
}

/*
 * Convert vart::RoundingMode to its symbolic string (e.g. "ROUND_TO_NEAREST_EVEN").
 */
static std::string get_rounding_mode_string(vart::RoundingMode mode) {
  switch (mode) {
    case vart::RoundingMode::ROUND_TO_NEAREST_EVEN:
      return "ROUND_TO_NEAREST_EVEN";
    case vart::RoundingMode::ROUND_TOWARD_ZERO:
      return "ROUND_TOWARD_ZERO";
    default:
      return "UNKNOWN";
  }
}

/*
 * @class pl_pass_through
 * @brief Thin native-XRT wrapper around the `pass_through` PL (Programmable
 *        Logic) kernel.
 *
 * The pass_through HLS kernel has the signature:
 *     void pass_through(const ap_uint<512>* in, ap_uint<512>* out, int size);
 * It copies `size` 512-bit words (64-byte AXI4 beats) from global-memory bank
 * gmem0 (in) to gmem1 (out). This wrapper loads the kernel from an .xclbin and
 * exposes forward():
 * copy `nbytes` from a host source buffer through the kernel and write the
 * kernel result to a host destination buffer. The transfer is byte-oriented,
 * so it is agnostic to the tensor data type (int8 / bf16 / fp32 / ...).
 */
class pl_pass_through {
 public:
  /* The pass_through kernel has a 512-bit (64-byte) AXI4 data path: its `size`
   * argument is the number of 512-bit words (beats) to copy, and each m_axi
   * access is a full 64-byte beat. Host transfers are therefore rounded up to a
   * whole beat, and the kernel is passed the beat count (not a 32-bit word
   * count). */
  static constexpr size_t kBeatBytes = 64;

  pl_pass_through(const std::string& xclbin_path,
                  const std::string& kernel_name,
                  AppLogLevel app_log,
                  unsigned int device_index = 0)
      : m_app_log(app_log) {
    APP_LOG(AppLogLevel::INFO, m_app_log, "Opening XRT device %u for PL kernel '%s'", device_index,
            kernel_name.c_str());
    m_device = xrt::device(device_index);

    APP_LOG(AppLogLevel::INFO, m_app_log, "Loading xclbin: %s", xclbin_path.c_str());
    xrt::xclbin xclbin(xclbin_path);
    xrt::uuid uuid = m_device.register_xclbin(xclbin);

    /* A hw_context binds the registered xclbin; the kernel is created from it. */
    xrt::hw_context ctx(m_device, uuid);
    m_kernel = xrt::kernel(ctx, kernel_name);
    APP_LOG(AppLogLevel::INFO, m_app_log, "PL kernel '%s' ready", kernel_name.c_str());
  }

  /**
   * @brief Forward `nbytes` from src through the pass_through kernel, writing
   *        the kernel output to dst. src and dst may alias (in-place forward).
   */
  void forward(const void* src, void* dst, size_t nbytes) {
    if (nbytes == 0) {
      return;
    }
    ensure_capacity(nbytes);

    /* pass_through operates on 512-bit (64-byte) beats; round the byte count up
     * to a whole beat and pass the beat count as the kernel `size` argument. */
    const size_t nbeats = (nbytes + kBeatBytes - 1) / kBeatBytes;
    const size_t padded = nbeats * kBeatBytes;

    /* Stage 1 (to-PL): stage the source into the input bo and sync it to the
     * device. This is exactly the work eliminated by the zero-copy path. */
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto* in_host = m_in_bo.map<uint8_t*>();
    std::memcpy(in_host, src, nbytes);
    if (padded > nbytes) {
      std::memset(in_host + nbytes, 0, padded - nbytes); /* zero the tail beat */
    }
    m_in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    /* Stage 2 (PL exec): launch the kernel and wait for completion. Argument
     * order matches the kernel signature: (in, out, size). */
    const auto t1 = std::chrono::high_resolution_clock::now();
    auto run = m_kernel(m_in_bo, m_out_bo, static_cast<int>(nbeats));
    run.wait();

    /* Stage 3 (from-PL): sync the result back and copy it to dst. */
    const auto t2 = std::chrono::high_resolution_clock::now();
    m_out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(dst, m_out_bo.map<uint8_t*>(), nbytes);
    const auto t3 = std::chrono::high_resolution_clock::now();

    m_to_pl_us += us(t0, t1);
    m_pl_exec_us += us(t1, t2);
    m_from_pl_us += us(t2, t3);
  }

  /**
   * @brief Import an NPU output buffer (exported as a dma-buf fd) into this
   *        PL device as an xrt::bo. The returned bo shares the same physical
   *        DDR memory as the NPU output tensor, so the PL kernel can read the
   *        inference result directly with no host copy or host->device sync.
   *
   * @param dmabuf_fd dma-buf file descriptor from NpuTensor::export_buffer().
   * @return An xrt::bo bound to this PL device, aliasing the exported buffer.
   */
  xrt::bo import_input(int dmabuf_fd) {
    return xrt::bo(m_device, static_cast<xrt::bo::export_handle>(dmabuf_fd));
  }

  /**
   * @brief Allocate a persistent PL output bo (gmem1) sized to nbytes (rounded
   *        up to whole 512-bit beats). The caller keeps the bo alive and reads
   *        the kernel result directly from its host mapping (bo.map()), so no
   *        host copy of the output is needed — this enables output-side
   *        zero-copy in forward_zerocopy_io().
   *
   * @param nbytes Number of valid bytes the kernel will write to this bo.
   * @return A device bo bound to gmem1, host-mappable via bo.map().
   */
  xrt::bo alloc_output(size_t nbytes) {
    const size_t need = ((nbytes + kBeatBytes - 1) / kBeatBytes) * kBeatBytes;
    return xrt::bo(m_device, need, m_kernel.group_id(1)); /* gmem1 */
  }

  /**
   * @brief Full zero-copy I/O forward: the kernel reads the imported NPU-output
   *        bo in place (input zero-copy) and writes into a caller-owned,
   *        persistent output bo. Input and output are DIFFERENT buffers.
   *
   * Unlike forward(), there is no host memcpy on either side: the input is the
   * NPU output buffer read in place, and the host reads the result straight
   * from out_bo.map() (see alloc_output()). Only a FROM_DEVICE cache sync is
   * done so the CPU sees the kernel's writes; there is no from-PL memcpy.
   *
   * @param in_bo  Imported input bo (from import_input()); aliases NPU output.
   * @param out_bo Persistent PL output bo (from alloc_output()); host-mapped.
   * @param nbytes Number of valid bytes to forward.
   */
  void forward_zerocopy_io(const xrt::bo& in_bo, xrt::bo& out_bo, size_t nbytes) {
    if (nbytes == 0) {
      return;
    }

    /* pass_through operates on 512-bit (64-byte) beats; round the byte count up
     * to a whole beat and pass the beat count as the kernel `size` argument. */
    const size_t nbeats = (nbytes + kBeatBytes - 1) / kBeatBytes;

    /* No to-PL stage: the kernel reads the NPU output buffer in place. */
    /* Stage 2 (PL exec): launch the kernel and wait. */
    const auto t1 = std::chrono::high_resolution_clock::now();
    auto run = m_kernel(in_bo, out_bo, static_cast<int>(nbeats));
    run.wait();

    /* Stage 3 (from-PL): only a cache sync — the host reads out_bo.map()
     * directly, so there is no output memcpy. */
    const auto t2 = std::chrono::high_resolution_clock::now();
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto t3 = std::chrono::high_resolution_clock::now();

    /* to-PL stage is zero by construction for the zero-copy path. */
    m_pl_exec_us += us(t1, t2);
    m_from_pl_us += us(t2, t3);
  }

  /* Accumulated sub-stage timers (microseconds), summed over every forward()
   * / forward_zerocopy_io() call since the last reset_timers():
   *   to_pl_us   : host->PL input staging  (memcpy-in + sync TO_DEVICE); 0 for zero-copy
   *   pl_exec_us : pass_through kernel launch + wait (the PL dummy post processing)
   *   from_pl_us : PL->host output; host-copy = sync FROM_DEVICE + memcpy-out,
   *                zero-copy = sync FROM_DEVICE only (no memcpy) */
  double to_pl_us() const { return m_to_pl_us; }
  double pl_exec_us() const { return m_pl_exec_us; }
  double from_pl_us() const { return m_from_pl_us; }
  void reset_timers() {
    m_to_pl_us = 0.0;
    m_pl_exec_us = 0.0;
    m_from_pl_us = 0.0;
  }

 private:
  /* Elapsed microseconds between two high_resolution_clock time points. */
  static double us(const std::chrono::high_resolution_clock::time_point& a,
                   const std::chrono::high_resolution_clock::time_point& b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
  }

  /* (Re)allocate the device buffers when a larger transfer is requested.
   * Buffers are sized to whole 512-bit beats and reused across calls. */
  void ensure_capacity(size_t nbytes) {
    const size_t need = ((nbytes + kBeatBytes - 1) / kBeatBytes) * kBeatBytes;
    if (need <= m_capacity) {
      return;
    }
    m_in_bo = xrt::bo(m_device, need, m_kernel.group_id(0));  /* gmem0 */
    m_out_bo = xrt::bo(m_device, need, m_kernel.group_id(1)); /* gmem1 */
    m_capacity = need;
    APP_LOG(AppLogLevel::DEBUG, m_app_log, "Allocated PL buffers of %zu bytes", need);
  }

  xrt::device m_device;
  xrt::kernel m_kernel;
  xrt::bo m_in_bo;
  xrt::bo m_out_bo;
  size_t m_capacity = 0;
  AppLogLevel m_app_log;

  /* Per-stage timing accumulators (see the getters above). */
  double m_to_pl_us = 0.0;
  double m_pl_exec_us = 0.0;
  double m_from_pl_us = 0.0;
}; /* pl_pass_through */

/*
 * @class app_context
 * @brief Class to hold the application context for vart application
 */
class app_context {
 private:
  size_t m_batch_size;

  /* application options */
  app_opt m_app_opt;

  /* runner options */
  runner_opt m_runner_opt;

  /* input Tensor Buffers */
  size_t m_num_input_tensors;
  std::vector<std::vector<vart::NpuTensor>> m_inputs;
  std::vector<vart::NpuTensorInfo> m_input_tensors_info = {};

  /* output tensor buffers */
  size_t m_num_output_tensors;
  std::vector<std::vector<vart::NpuTensor>> m_outputs;
  std::vector<vart::NpuTensorInfo> m_output_tensors_info = {};

  /* output file handles */
  std::vector<std::fstream> m_output_files;
  std::vector<std::filesystem::path> m_output_file_paths;

  /* vart runner */
  std::shared_ptr<vart::Runner> m_runner = nullptr;

  /* pass_through PL kernel wrapper (native XRT). Created in init_pl_kernel()
   * for the inference flow; stays null in dry-run / get-model-info. */
  std::unique_ptr<pl_pass_through> m_pl = nullptr;

  /* Zero-copy input bos for the PL kernel: m_pl_in_bos[i][j] is the NPU output
   * tensor m_outputs[i][j] exported as a dma-buf and imported into the PL
   * device, so the kernel reads the inference result with no host copy. Built
   * once in setup_pl_zerocopy() and reused across all frames. */
  std::vector<std::vector<xrt::bo>> m_pl_in_bos;

  /* Zero-copy output bos for the PL kernel: m_pl_out_bos[i][j] is a persistent
   * PL output buffer (gmem1) the kernel writes into (different buffer from the
   * input), and m_pl_out_ptrs[i][j] is its host mapping. The host reads the
   * kernel result directly from that pointer with no from-PL memcpy. Built once
   * in setup_pl_zerocopy() alongside the input bos and reused across frames. */
  std::vector<std::vector<xrt::bo>> m_pl_out_bos;
  std::vector<std::vector<uint8_t*>> m_pl_out_ptrs;

  /* When true (default), the ML<->PL transfer uses the zero-copy path
   * (forward_zerocopy_io: dma-buf input + host-mapped output bo). Set to false
   * via env PL_ZEROCOPY=0 to use the original host-copy path (forward) for A/B
   * timing comparison. */
  bool m_pl_zerocopy = true;

  /* Number of frames the PL forward has covered (summed across all runs), used
   * as the denominator for the per-stage per-frame timing report. The per-stage
   * microsecond accumulators live inside the pass_through wrapper (m_pl). */
  size_t m_pl_forward_frames = 0;

  /* IFM file paths resolved in runner-reported input-tensor order.
   * Populated by resolve_ifms_in_runner_order() after the runner is
   * created; index j matches m_input_tensors_info[j]. */
  std::vector<std::string> m_ifm_files;

  /* tensor types */
  vart::TensorType m_input_tensor_type;
  vart::TensorType m_output_tensor_type;

  /* quantization parameters */
  std::unordered_map<std::string, vart::QuantParameters> m_quant_params = {};

  /**
   * @brief Clear tensors allocated by the runner (RAII handles deallocation)
   */
  void deallocate_tensors() {
    /* Runner-allocated tensors are automatically released via RAII */
    m_inputs.clear();
    m_outputs.clear();
  }

  /**
   * @brief Clear the application context
   */
  void clear() {
    m_num_input_tensors = 0;
    m_num_output_tensors = 0;
    m_input_tensors_info.clear();
    m_output_tensors_info.clear();
    m_quant_params.clear();
    m_batch_size = 1;

    /* Deallocate tensors allocated by runner */
    deallocate_tensors();
    m_runner = nullptr;
    m_pl = nullptr;
    m_ifm_files.clear();
  }

  /**
   * @brief Print tensor metadata
   */
  void print_tensor_metadata() {
    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Number of input tensors: %zu", m_num_input_tensors);
    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Number of output tensors: %zu", m_num_output_tensors);
    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Batch size: %zu", m_batch_size);

    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Input Tensors Info:");
    for (size_t i = 0; i < m_input_tensors_info.size(); ++i) {
      cout << "Tensor " << i << ":" << std::endl;
      m_input_tensors_info[i].print();
    }

    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Output Tensors Info:");
    for (size_t i = 0; i < m_output_tensors_info.size(); ++i) {
      cout << "Tensor " << i << ":" << std::endl;
      m_output_tensors_info[i].print();
    }

    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Quantization Parameters:");
    for (const auto& [name, params] : m_quant_params) {
      APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Tensor: %s, Scale: %f, Zero Point: %d, Rounding Mode: %s",
              name.c_str(), params.scale, params.zero_point, get_rounding_mode_string(params.rounding_mode).c_str());
    }
  }

 public:
  /**
   * @brief Constructor to initialize the application context
   * @param app_options Application options
   * @param runner_options Runner options
   */
  app_context(const app_opt& app_options, const runner_opt& runner_options)
      : m_app_opt(app_options),
        m_runner_opt(runner_options),
        m_input_tensor_type(runner_options.input_tensor_type == "CPU" ? vart::TensorType::CPU : vart::TensorType::HW),
        m_output_tensor_type(runner_options.output_tensor_type == "CPU" ? vart::TensorType::CPU
                                                                        : vart::TensorType::HW) {
    /* initialize member variables */
    clear();
  }

  /**
   * @brief Destructor to clean up the application context
   */
  ~app_context() { clear(); }

  /**
   * @brief Get tensor metadata from the runner
   * @return Status of the operation
   */
  vart_app_status get_tensor_metadata() {
    try {
      /* Get tensors metadata */
      m_num_input_tensors = m_runner->get_num_input_tensors();
      m_num_output_tensors = m_runner->get_num_output_tensors();
      m_batch_size = m_runner->get_batch_size();

      /* Get Tensor Information based on configured tensor types */
      m_input_tensors_info = m_runner->get_tensors_info(vart::TensorDirection::INPUT, m_input_tensor_type);
      m_output_tensors_info = m_runner->get_tensors_info(vart::TensorDirection::OUTPUT, m_output_tensor_type);

      /* Querying quantization parameters for HW input tensors */
      for (const auto& tensor_info : m_input_tensors_info) {
        m_quant_params[tensor_info.name] = m_runner->get_quant_parameters(tensor_info.name);
      }

      /* Querying quantization parameters for HW output tensors */
      for (const auto& tensor_info : m_output_tensors_info) {
        m_quant_params[tensor_info.name] = m_runner->get_quant_parameters(tensor_info.name);
      }
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error getting tensor metadata: %s", e.what());
      return vart_app_status::FAILURE;
    }

    /* Print the loaded metadata at INFO+ log levels. Kept inside this
     * method so the load and the visibility-controlled print stay
     * together */
    if (m_app_opt.app_log >= AppLogLevel::INFO) {
      print_tensor_metadata();
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Bind ifms-config entries to the runner's input tensors by name.
   *
   * The user's ifms-config is order-independent. Each entry's `name` must
   * match a runner-reported input tensor name; this method validates the
   * one-to-one mapping and materialises @c m_ifm_files in runner-reported
   * input-tensor order so downstream code can keep indexing positionally.
   *
   * Strict validation: any size mismatch or unknown tensor name aborts
   * with both lists (JSON names vs. runner-reported names) so the user
   * can edit the JSON without re-running --get-model-info.
   *
   * @return SUCCESS when every runner-reported input tensor has a file
   *         bound to it, FAILURE otherwise.
   */
  vart_app_status resolve_ifms_in_runner_order() {
    m_ifm_files.clear();

    const auto& by_name = m_app_opt.ifm_files_by_name;
    if (by_name.size() != m_num_input_tensors) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
              "ifms-config provides %zu entry/entries, but the model expects %zu input tensor(s).", by_name.size(),
              m_num_input_tensors);
      log_ifm_name_mismatch();
      return vart_app_status::FAILURE;
    }

    m_ifm_files.reserve(m_num_input_tensors);
    for (size_t j = 0; j < m_num_input_tensors; ++j) {
      const std::string& tname = m_input_tensors_info[j].name;
      auto it = by_name.find(tname);
      if (it == by_name.end()) {
        APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
                "ifms-config is missing the input for runner-expected tensor '%s'.", tname.c_str());
        log_ifm_name_mismatch();
        return vart_app_status::FAILURE;
      }
      m_ifm_files.push_back(it->second);
      APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Bound ifms-config name='%s' -> file='%s'", tname.c_str(),
              it->second.c_str());
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Emit both name lists (ifms-config and runner-reported) to aid
   *        the user in fixing an ifms-config / runner name mismatch.
   */
  void log_ifm_name_mismatch() {
    std::string json_names;
    for (const auto& kv : m_app_opt.ifm_files_by_name) {
      if (!json_names.empty())
        json_names += ", ";
      json_names += "'" + kv.first + "'";
    }
    std::string runner_names;
    for (const auto& info : m_input_tensors_info) {
      if (!runner_names.empty())
        runner_names += ", ";
      runner_names += "'" + info.name + "'";
    }
    APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "  ifms-config 'name' values   : [%s]", json_names.c_str());
    APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "  Runner-expected tensor names: [%s]", runner_names.c_str());
  }

  /**
   * @brief Query the runner for both HW and CPU tensor metadata, print a
   *        human-readable summary to the console, and dump the same content
   *        to `<model_basename>_info.json` in the current working directory.
   *
   * The dump is independent of the user-configured `input_tensor_type` and
   * `output_tensor_type` (those drive the inference path only). The JSON
   * always contains both an `"hw"` and a `"cpu"` view per direction so the
   * file fully describes what the model exposes. Tensors are emitted in
   * the order returned by the runner.
   *
   * @param output_path On success, the absolute path of the dumped JSON
   *                    file is written here.
   * @return SUCCESS on completion, FAILURE on any I/O or serialisation
   *         error.
   */
  vart_app_status dump_model_info_to_json(std::filesystem::path& output_path) {
    try {
      /* Query the runner directly for both HW and CPU views so the dump
       * is independent of the user-configured input/output tensor types.
       * CPU tensors are always reported by the runner; HW tensors may be
       * empty for models that do not expose a HW view, so the CPU vectors
       * drive iteration order and the HW view is merged when present. */
      auto hw_inputs = m_runner->get_tensors_info(vart::TensorDirection::INPUT, vart::TensorType::HW);
      auto cpu_inputs = m_runner->get_tensors_info(vart::TensorDirection::INPUT, vart::TensorType::CPU);
      auto hw_outputs = m_runner->get_tensors_info(vart::TensorDirection::OUTPUT, vart::TensorType::HW);
      auto cpu_outputs = m_runner->get_tensors_info(vart::TensorDirection::OUTPUT, vart::TensorType::CPU);

      /* Quantization parameters are queried per tensor name via
       * get_quant_parameters(name). The CPU vectors are the superset of
       * tensor names (HW views may be empty for tensors without a HW
       * counterpart), so iterate over cpu_inputs / cpu_outputs to make
       * sure every tensor in the model is covered. The same block applies
       * to both views of the same tensor when both views exist. */
      std::unordered_map<std::string, vart::QuantParameters> tensor_quant;
      for (const auto& info : cpu_inputs) {
        tensor_quant[info.name] = m_runner->get_quant_parameters(info.name);
      }
      for (const auto& info : cpu_outputs) {
        tensor_quant[info.name] = m_runner->get_quant_parameters(info.name);
      }

      /* Quantization parameters describe how integer tensor data maps to
       * real-valued data. A view whose dtype is a floating-point type
       * already carries real values, so emitting scale/zero_point there
       * would mislead the consumer into dequantizing again. Only emit the
       * block for views with integer dtypes. */
      auto is_quantized_dtype = [](vart::DataType t) {
        switch (t) {
          case vart::DataType::INT8:
          case vart::DataType::UINT8:
          case vart::DataType::INT16:
          case vart::DataType::UINT16:
          case vart::DataType::INT32:
          case vart::DataType::UINT32:
          case vart::DataType::INT64:
          case vart::DataType::UINT64:
            return true;
          default:
            return false;
        }
      };

      ptree root;
      root.put("model_file", m_runner_opt.model_cache_dir);
      root.put("batch_size", m_runner->get_batch_size());

      /* Build a name -> NpuTensorInfo lookup for the HW view so the merge
       * with the CPU view (driven from the CPU vector to preserve ordering
       * and to handle HW-empty cases) is O(1) per tensor. The runner
       * reports the same `name` for both views of the same tensor; if a
       * HW counterpart is missing we still emit the CPU-only fields. */
      auto build_index = [](const std::vector<vart::NpuTensorInfo>& infos) {
        std::unordered_map<std::string, const vart::NpuTensorInfo*> idx;
        idx.reserve(infos.size());
        for (const auto& info : infos)
          idx[info.name] = &info;
        return idx;
      };
      auto hw_input_idx = build_index(hw_inputs);
      auto hw_output_idx = build_index(hw_outputs);

      /* Serialise one view (HW or CPU) as a sub-object: shape array,
       * memory_layout/dtype strings, byte size, and quantization. Quant
       * parameters apply per tensor (same values for any integer view) and
       * are emitted only when this view's dtype is itself an integer. */
      auto serialise_view = [&tensor_quant, &is_quantized_dtype](const vart::NpuTensorInfo& info) {
        ptree v;
        ptree shape_array;
        for (const auto& dim : info.shape) {
          ptree d;
          d.put("", dim);
          shape_array.push_back(std::make_pair("", d));
        }
        v.add_child("shape", shape_array);
        v.put("memory_layout", get_memory_layout_string(info.memory_layout));
        /* For GENERIC memory layout, the memory_layout string itself
         * ("GENERIC") is not informative; the actual dimension permutation
         * lives in NpuTensorInfo::memory_layout_order (see
         * vart_npu_tensor.hpp). Emit it only when the layout is GENERIC and
         * the vector is populated, so other layouts keep their compact JSON
         * representation. */
        if (info.memory_layout == vart::MemoryLayout::GENERIC && !info.memory_layout_order.empty()) {
          ptree order_array;
          for (const auto& d : info.memory_layout_order) {
            ptree e;
            e.put("", d);
            order_array.push_back(std::make_pair("", e));
          }
          v.add_child("memory_layout_order", order_array);
        }
        v.put("dtype", get_data_type_string(info.data_type));
        v.put("size_in_bytes", info.size_in_bytes);

        if (is_quantized_dtype(info.data_type)) {
          auto qp_it = tensor_quant.find(info.name);
          if (qp_it != tensor_quant.end()) {
            ptree qp;
            qp.put("scale", qp_it->second.scale);
            qp.put("zero_point", qp_it->second.zero_point);
            qp.put("rounding_mode", get_rounding_mode_string(qp_it->second.rounding_mode));
            v.add_child("quantization", qp);
          }
        }
        return v;
      };

      /* Build a tensor entry: { "name": ..., "hw": {...}, "cpu": {...} }.
       * CPU is emitted first because it is always present; HW is appended
       * after it and is omitted when the runner does not expose a HW
       * view for this tensor name. */
      auto serialise_tensor = [&serialise_view](const vart::NpuTensorInfo& cpu_info,
                                                const vart::NpuTensorInfo* hw_info) {
        ptree t;
        t.put("name", cpu_info.name);
        t.add_child("cpu", serialise_view(cpu_info));
        if (hw_info != nullptr) {
          t.add_child("hw", serialise_view(*hw_info));
        }
        return t;
      };

      auto build_array = [&serialise_tensor](
                             const std::vector<vart::NpuTensorInfo>& cpu_infos,
                             const std::unordered_map<std::string, const vart::NpuTensorInfo*>& hw_idx) {
        ptree arr;
        for (const auto& cpu_info : cpu_infos) {
          auto it = hw_idx.find(cpu_info.name);
          const vart::NpuTensorInfo* hw_info = (it != hw_idx.end()) ? it->second : nullptr;
          arr.push_back(std::make_pair("", serialise_tensor(cpu_info, hw_info)));
        }
        return arr;
      };

      root.add_child("inputs", build_array(cpu_inputs, hw_input_idx));
      root.add_child("outputs", build_array(cpu_outputs, hw_output_idx));

      /* Console summary - written to stdout regardless of --log-level so
       * the operator sees the dump even with default logging. */
      auto print_tensor_list = [&tensor_quant, &is_quantized_dtype](
                                   const std::string& label, const std::vector<vart::NpuTensorInfo>& cpu_infos,
                                   const std::unordered_map<std::string, const vart::NpuTensorInfo*>& hw_idx) {
        /* Generic over the shape element type because vart::NpuTensorInfo::shape
         * is currently std::vector<uint32_t>; using auto here keeps the lambda
         * working if VART later switches to int64_t or any integral type. */
        auto print_shape = [](const auto& shape) {
          std::cout << "[";
          for (size_t d = 0; d < shape.size(); ++d) {
            if (d)
              std::cout << ",";
            std::cout << shape[d];
          }
          std::cout << "]";
        };
        /* For GENERIC memory layout the symbolic name carries no shape info,
         * so append the dimension permutation captured in
         * NpuTensorInfo::memory_layout_order. */
        auto print_layout = [&](const vart::NpuTensorInfo& info) {
          std::cout << "  memory_layout=" << get_memory_layout_string(info.memory_layout);
          if (info.memory_layout == vart::MemoryLayout::GENERIC && !info.memory_layout_order.empty()) {
            std::cout << "(memory_layout_order=";
            print_shape(info.memory_layout_order);
            std::cout << ")";
          }
        };
        /* Quantization is per tensor but only meaningful for views whose
         * dtype is an integer type; floating-point views already carry
         * real-valued data. */
        auto print_quant = [&](const vart::NpuTensorInfo& info) {
          if (!is_quantized_dtype(info.data_type))
            return;
          auto qp_it = tensor_quant.find(info.name);
          if (qp_it != tensor_quant.end()) {
            std::cout << "  quant{scale=" << qp_it->second.scale << ", zero_point=" << qp_it->second.zero_point
                      << ", rounding_mode=" << get_rounding_mode_string(qp_it->second.rounding_mode) << "}";
          }
        };
        std::cout << "  " << label << " (" << cpu_infos.size() << "):\n";
        for (size_t i = 0; i < cpu_infos.size(); ++i) {
          const auto& cpu = cpu_infos[i];
          std::cout << "    [" << i << "] " << cpu.name << "\n";
          std::cout << "         cpu: shape=";
          print_shape(cpu.shape);
          std::cout << "  dtype=" << get_data_type_string(cpu.data_type);
          print_layout(cpu);
          std::cout << "  size=" << cpu.size_in_bytes << "B";
          print_quant(cpu);
          std::cout << "\n";
          auto hit = hw_idx.find(cpu.name);
          if (hit != hw_idx.end()) {
            const auto& hw = *hit->second;
            std::cout << "         hw : shape=";
            print_shape(hw.shape);
            std::cout << "  dtype=" << get_data_type_string(hw.data_type);
            print_layout(hw);
            std::cout << "  size=" << hw.size_in_bytes << "B";
            print_quant(hw);
            std::cout << "\n";
          }
        }
      };

      std::cout << "\n--- Model info ---\n";
      std::cout << "Model file        : " << m_runner_opt.model_cache_dir << "\n";
      std::cout << "Batch size : " << m_runner->get_batch_size() << "\n";
      print_tensor_list("Inputs", cpu_inputs, hw_input_idx);
      std::cout << "\n";
      print_tensor_list("Outputs", cpu_outputs, hw_output_idx);
      std::cout << "\n";

      /* Resolve the output filename from the model_cache_dir basename so a
       * caller running multiple models back-to-back gets one JSON per
       * model. The model path can be either a .rai file or a directory
       * (with or without a trailing slash, e.g. "foo/ResNet50/"). For
       * directories we use the directory name; for files we use the stem
       * (filename without extension).
       *
       * We strip trailing separators from the raw string explicitly
       * because std::filesystem::path::lexically_normal() is
       * implementation-defined for that case (libstdc++ leaves a trailing
       * empty filename, libc++ removes the separator), which made the
       * earlier path-API-only approach return an empty basename on the
       * board. */
      std::string raw = m_runner_opt.model_cache_dir;
      while (!raw.empty() && (raw.back() == '/' || raw.back() == '\\')) {
        raw.pop_back();
      }
      std::filesystem::path model_path(raw);
      std::string basename;
      if (std::filesystem::is_directory(model_path)) {
        basename = model_path.filename().string();
      } else {
        basename = model_path.stem().string();
      }
      if (basename.empty())
        basename = "model";
      output_path = std::filesystem::current_path() / (basename + "_info.json");

      /* Boost.PropertyTree's JSON writer escapes the forward slashes inside
       * string values (a JSON-spec-legal but visually noisy choice). Serialise
       * to a string first, strip the `\/` escapes, then write the cleaned text
       * to disk so paths like "model_cache_dir/foo/bar.rai" stay readable. */
      std::ostringstream oss;
      boost::property_tree::write_json(oss, root);
      std::string serialised = oss.str();
      /* Boost.PropertyTree escapes '/' as "\\/" in its JSON output. Replace
       * each occurrence with a bare '/'. Resuming the next search from `pos`
       * (now the just-written '/') is intentional: it advances exactly one
       * character, so the loop is guaranteed to make progress without
       * needing `pos + 1`. */
      for (size_t pos = serialised.find("\\/"); pos != std::string::npos; pos = serialised.find("\\/", pos)) {
        serialised.replace(pos, 2, "/");
      }
      std::ofstream out(output_path);
      if (!out.is_open()) {
        APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to open output file for model info: %s",
                output_path.string().c_str());
        return vart_app_status::FAILURE;
      }
      out << serialised;
      std::cout << "Model info written to: " << output_path.string() << "\n"
                << "\n";
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to dump model info JSON: %s", e.what());
      return vart_app_status::FAILURE;
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Calculate total number of frames in the input files
   * @return Total number of frames
   */
  size_t calculate_ifms_total_frames() {
    if (m_ifm_files.size() != m_num_input_tensors) {
      throw std::invalid_argument("Mismatch between number of resolved IFM files and tensor sizes.");
    }

    size_t total_frames = 0;

    for (size_t j = 0; j < m_num_input_tensors; ++j) {
      auto ifm_path = std::filesystem::path(m_ifm_files[j]);

      // Check if the file exists
      if (!std::filesystem::exists(ifm_path)) {
        throw std::runtime_error("Input file does not exist: " + m_ifm_files[j]);
      }

      // Get the file size
      auto file_size = std::filesystem::file_size(ifm_path);

      // Calculate the number of frames for this tensor
      size_t frames = file_size / m_input_tensors_info[j].size_in_bytes;

      // Check if the file size is perfectly divisible by the tensor size
      if (file_size % m_input_tensors_info[j].size_in_bytes != 0) {
        throw std::runtime_error("File size is not divisible by tensor size for file: " + m_ifm_files[j]);
      }

      // For the first tensor, initialize total_frames
      if (j == 0) {
        total_frames = frames;
      } else {
        // Check if the number of frames is consistent across all tensors
        if (frames != total_frames) {
          throw std::runtime_error("Inconsistent number of frames across tensors.");
        }
      }
    }

    return total_frames;
  }

  /**
   * @brief Allocates memory for input tensors required for inference using Runner.
   * @return Returns 0 on successful allocation, or an error code on failure.
   */
  int allocate_input_tensors() {
    m_inputs.resize(m_batch_size);
    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Allocating input tensors: batch_size=%zu, num_tensors=%zu",
            m_batch_size, m_num_input_tensors);

    for (size_t i = 0; i < m_batch_size; ++i) {
      m_inputs[i].reserve(m_num_input_tensors);
      for (size_t j = 0; j < m_num_input_tensors; ++j) {
        try {
          vart::NpuTensor input_tensor = m_runner->allocate_npu_tensor(m_input_tensors_info[j]);
          APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Allocated input tensor[%zu][%zu] of size: %zu bytes", i, j,
                  m_input_tensors_info[j].size_in_bytes);
          m_inputs[i].push_back(std::move(input_tensor));
        } catch (const std::exception& e) {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
                  "Failed to allocate input tensor for batch[%zu], tensor[%zu]: %s", i, j, e.what());
          return -1;
        }
      }
    }

    APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Input tensors allocated successfully.");
    return 0;
  }

  /**
   * @brief Load data into pre-allocated input tensors
   * @param frame_count The current frame count to determine the offset for loading data
   * @param actual_batch_size The actual number of frames to process
   * @return Status of the operation
   */
  vart_app_status populate_input_tensors(size_t frame_count, size_t actual_batch_size) {
    for (size_t i = 0; i < actual_batch_size; ++i) {
      for (size_t j = 0; j < m_num_input_tensors; ++j) {
        auto tensor_size_in_bytes = m_input_tensors_info[j].size_in_bytes;

        /* Calculate the offset based on frame_count + batch position */
        size_t offset = (frame_count + i) * tensor_size_in_bytes;

        /* Get virtual address from pre-allocated tensor */
        void* virtual_addr = m_inputs[i][j].get_virtual_address();
        if (!virtual_addr) {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to get virtual address for input tensor[%zu][%zu]", i,
                  j);
          return vart_app_status::FAILURE;
        }

        /* Load input data from file at the calculated offset */
        auto ifm_path = (fs::path(m_ifm_files[j]));
        if (!utils::load_binary_data(ifm_path, virtual_addr, tensor_size_in_bytes, offset)) {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to load input data from %s at offset: %zu",
                  ifm_path.c_str(), offset);
          return vart_app_status::FAILURE;
        }

        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log,
                "Loaded input data for batch[%zu] from %s of size: %zu at offset: %zu", i, ifm_path.c_str(),
                tensor_size_in_bytes, offset);
      }
    }

    APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Input tensors populated successfully for actual_batch_size=%zu.",
            actual_batch_size);
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Allocate memory buffers for output tensors using Runner.
   * @return Status of the operation
   */
  int allocate_output_tensors() {
    m_outputs.resize(m_batch_size);
    APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Allocating output tensors: batch_size=%zu, num_tensors=%zu",
            m_batch_size, m_num_output_tensors);

    for (size_t i = 0; i < m_batch_size; ++i) {
      m_outputs[i].reserve(m_num_output_tensors);
      for (size_t j = 0; j < m_num_output_tensors; ++j) {
        try {
          vart::NpuTensor output_tensor = m_runner->allocate_npu_tensor(m_output_tensors_info[j]);
          APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Allocated output tensor[%zu][%zu] of size: %zu bytes", i, j,
                  m_output_tensors_info[j].size_in_bytes);
          m_outputs[i].push_back(std::move(output_tensor));
        } catch (const std::exception& e) {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
                  "Failed to allocate output tensor for batch[%zu], tensor[%zu]: %s", i, j, e.what());
          return -1;
        }
      }
    }

    APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Output tensors allocated successfully.");
    return 0;
  }

  /*
   * @brief Create the VART runner
   * @return Status of the operation
   */
  vart_app_status create_runner() {
    try {
      /* Create VART Runner */
      APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Creating VART Runner...");
      APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log,
              "Runner options - log_level: %s, input_tensor_type: %s, output_tensor_type: %s, aie_columns_sharing: %s, "
              "ai_analyzer_profiling: %s",
              m_runner_opt.log_level.c_str(), m_runner_opt.input_tensor_type.c_str(),
              m_runner_opt.output_tensor_type.c_str(), m_runner_opt.aie_columns_sharing ? "true" : "false",
              m_runner_opt.ai_analyzer_profiling ? "true" : "false");

      /* vart runner options */
      std::unordered_map<std::string, std::any> options = {
          {"log_level", m_runner_opt.log_level},
          {"input_tensor_type", m_runner_opt.input_tensor_type},
          {"output_tensor_type", m_runner_opt.output_tensor_type},
          {"aie_columns_sharing", m_runner_opt.aie_columns_sharing},
          {"ai_analyzer_profiling", m_runner_opt.ai_analyzer_profiling}};

      /* Add config_json if specified */
      if (!m_runner_opt.config_json.empty()) {
        if (fs::exists(m_runner_opt.config_json)) {
          APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Using Vitis AI config file: %s",
                  m_runner_opt.config_json.c_str());
        } else {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Vitis AI config file not found: %s",
                  m_runner_opt.config_json.c_str());
          return vart_app_status::FAILURE;
        }
        options["config_json"] = m_runner_opt.config_json;
      }

      /* Add start_column if explicitly set */
      if (m_runner_opt.is_start_column_set) {
        options["start_column"] = m_runner_opt.start_column;
      }

      /* Add cma_index if explicitly set */
      if (m_runner_opt.is_cma_index_set) {
        options["cma_index"] = m_runner_opt.cma_index;
      }

      /* Create Runner */
      try {
        m_runner = vart::RunnerFactory::create_runner(vart::RunnerType::VAIML, m_runner_opt.model_cache_dir, options);
      } catch (const std::exception& e) {
        APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error creating runner: %s", e.what());
        return vart_app_status::FAILURE;
      }

      return vart_app_status::SUCCESS;
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error creating VART runner: %s", e.what());
      return vart_app_status::FAILURE;
    }
  }

  /**
   * @brief Run inference on the prepared input tensors
   * @return Status of the operation
   */
  vart_app_status run_inference() {
    try {
      /*
       * Runs inference on the prepared input tensors and fills the output tensors.
       */
      auto ret = m_runner->execute(m_inputs, m_outputs);
      if (vart::StatusCode::SUCCESS == ret) {
        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Inference executed successfully");
        return vart_app_status::SUCCESS;
      } else {
        APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Inference failed with error code: %d", static_cast<int>(ret));
        return vart_app_status::FAILURE;
      }
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error during inference: %s", e.what());
      return vart_app_status::FAILURE;
    }
  }

  /**
   * @brief Create the pass_through PL kernel wrapper from app options.
   *
   * Mandatory for the inference flow: this application forwards every
   * inference output tensor through the PL kernel and uses the kernel output
   * as the application output. Skipped in dry-run (no real data to forward).
   * @return Status of the operation.
   */
  vart_app_status init_pl_kernel() {
    if (m_app_opt.dry_run) {
      APP_LOG(AppLogLevel::INFO, m_app_opt.app_log, "Dry-run: skipping PL kernel initialization.");
      return vart_app_status::SUCCESS;
    }
    if (m_app_opt.pl_xclbin.empty()) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
              "Missing 'pl-config.xclbin-location': the .xclbin containing the '%s' PL kernel is required.",
              m_app_opt.pl_kernel.c_str());
      return vart_app_status::FAILURE;
    }
    if (!fs::exists(m_app_opt.pl_xclbin)) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "xclbin not found: %s", m_app_opt.pl_xclbin.c_str());
      return vart_app_status::FAILURE;
    }
    try {
      m_pl = std::make_unique<pl_pass_through>(m_app_opt.pl_xclbin, m_app_opt.pl_kernel, m_app_opt.app_log,
                                               m_app_opt.pl_device_index);
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to initialize PL kernel: %s", e.what());
      return vart_app_status::FAILURE;
    }
    /* Select the ML->PL transfer path. Zero-copy (dma-buf) is the default;
     * PL_ZEROCOPY=0 forces the original host-copy path for A/B comparison. */
    if (const char* env = std::getenv("PL_ZEROCOPY")) {
      m_pl_zerocopy = !(env[0] == '0' && env[1] == '\0');
    }
    APP_LOG(AppLogLevel::RESULT, m_app_opt.app_log, "ML->PL transfer mode: %s",
            m_pl_zerocopy ? "zero-copy (dma-buf)" : "host-copy");
    if (m_pl_zerocopy) {
      /* Bind the runner-allocated output tensors to the PL device via dma-buf
       * so the kernel reads them zero-copy. Output tensors are allocated once
       * and reused, so this is done once here. */
      return setup_pl_zerocopy();
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Export every inference output NpuTensor as a dma-buf and import it
   *        into the PL device, so forward_outputs_through_pl() can run the
   *        pass_through kernel directly on the NPU output buffers (zero-copy),
   *        eliminating the per-frame host copy and host->device DMA of the
   *        ML->PL transfer.
   *
   * The output tensors (m_outputs) are runner-allocated once and reused across
   * all frames, so the exported/imported bos are also built once and reused.
   * @return Status of the operation.
   */
  vart_app_status setup_pl_zerocopy() {
    if (!m_pl) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "PL kernel not initialized before zero-copy setup.");
      return vart_app_status::FAILURE;
    }
    try {
      m_pl_in_bos.clear();
      m_pl_in_bos.resize(m_batch_size);
      m_pl_out_bos.clear();
      m_pl_out_bos.resize(m_batch_size);
      m_pl_out_ptrs.clear();
      m_pl_out_ptrs.resize(m_batch_size);
      for (size_t i = 0; i < m_batch_size; ++i) {
        m_pl_in_bos[i].reserve(m_num_output_tensors);
        m_pl_out_bos[i].reserve(m_num_output_tensors);
        m_pl_out_ptrs[i].reserve(m_num_output_tensors);
        for (size_t j = 0; j < m_num_output_tensors; ++j) {
          /* Input side (zero-copy): import the NPU output tensor as a dma-buf
           * so the kernel reads it in place. */
          const int fd = m_outputs[i][j].export_buffer();
          if (fd < 0) {
            APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
                    "Failed to export dma-buf for output tensor[%zu][%zu]", i, j);
            return vart_app_status::FAILURE;
          }
          m_pl_in_bos[i].push_back(m_pl->import_input(fd));

          /* Output side (zero-copy): a persistent PL output bo (different
           * buffer from the input) the kernel writes into, mapped once so the
           * host reads the result directly with no from-PL memcpy. */
          xrt::bo out_bo = m_pl->alloc_output(m_output_tensors_info[j].size_in_bytes);
          m_pl_out_ptrs[i].push_back(out_bo.map<uint8_t*>());
          m_pl_out_bos[i].push_back(std::move(out_bo));

          APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log,
                  "Bound output tensor[%zu][%zu] for zero-copy I/O (in imported, out mapped)", i, j);
        }
      }
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error setting up PL zero-copy buffers: %s", e.what());
      return vart_app_status::FAILURE;
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Forward this batch's inference outputs through the pass_through PL
   *        kernel.
   *
   * For each valid frame and each output tensor the kernel is launched on the
   * inference result. In zero-copy mode (default) the kernel reads the NPU
   * output buffer in place and writes into a persistent PL output bo, and the
   * kernel result is read later from that bo's host mapping (m_pl_out_ptrs) in
   * write_output_tensors() — no host copy on either side. In the host-copy
   * path the result is staged to the PL device and written back over the NPU
   * output tensor. For the pass_through kernel the output is a byte-exact copy
   * of the inference result in both modes.
   *
   * @param actual_batch_size Number of valid frames in the current batch.
   * @return Status of the operation.
   */
  vart_app_status forward_outputs_through_pl(size_t actual_batch_size) {
    if (!m_pl) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "PL kernel not initialized.");
      return vart_app_status::FAILURE;
    }
    try {
      for (size_t i = 0; i < actual_batch_size; ++i) {
        for (size_t j = 0; j < m_num_output_tensors; ++j) {
          const size_t nbytes = m_output_tensors_info[j].size_in_bytes;
          void* data_ptr = m_outputs[i][j].get_virtual_address();
          if (!data_ptr) {
            APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
                    "Failed to get virtual address for output tensor[%zu][%zu]", i, j);
            return vart_app_status::FAILURE;
          }
          if (m_pl_zerocopy) {
            /* Zero-copy I/O: the kernel reads the NPU output buffer in place
             * (imported via dma-buf) and writes into a persistent PL output bo.
             * No host copy or host->device sync on the input side and no
             * from-PL memcpy on the output side; the host reads the result
             * directly from m_pl_out_ptrs[i][j] (see write_output_tensors). */
            (void)data_ptr;
            m_pl->forward_zerocopy_io(m_pl_in_bos[i][j], m_pl_out_bos[i][j], nbytes);
          } else {
            /* Host-copy ML->PL: memcpy the inference result into a staging bo,
             * sync host->device, run the kernel, sync back, memcpy out. */
            m_pl->forward(data_ptr, data_ptr, nbytes);
          }
          APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log,
                  "Forwarded output tensor[%zu][%zu] (%zu bytes) through PL kernel (%s)", i, j, nbytes,
                  m_pl_zerocopy ? "zero-copy" : "host-copy");
        }
      }
      /* Fine-grained per-stage timing is accumulated inside the pass_through
       * wrapper (forward/forward_zerocopy_io); here we only track how many
       * frames the PL forward covered, for the per-frame averages. */
      m_pl_forward_frames += actual_batch_size;
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error forwarding outputs through PL kernel: %s", e.what());
      return vart_app_status::FAILURE;
    }
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Convert a VART DataType to its string representation.
   * @param data_type The VART DataType enumeration value to be converted.
   * @return A string representing the given data type.
   */
  std::string dataTypeToString(vart::DataType data_type) {
    switch (data_type) {
      case vart::DataType::BOOLEAN:
        return "boolean";
      case vart::DataType::INT8:
        return "int8";
      case vart::DataType::UINT8:
        return "uint8";
      case vart::DataType::INT16:
        return "int16";
      case vart::DataType::UINT16:
        return "uint16";
      case vart::DataType::BF16:
        return "bf16";
      case vart::DataType::FP16:
        return "fp16";
      case vart::DataType::INT32:
        return "int32";
      case vart::DataType::UINT32:
        return "uint32";
      case vart::DataType::FLOAT32:
        return "float32";
      case vart::DataType::INT64:
        return "int64";
      case vart::DataType::UINT64:
        return "uint64";
      case vart::DataType::UNKNOWN:
        throw std::runtime_error("Error: DataType is UNKNOWN");
      default:
        throw std::runtime_error("Error: Unrecognized DataType");
    }
  }

  /**
   * @brief Construct output filename for a tensor
   * @param iteration Current iteration number
   * @param tensor_idx Tensor index
   * @param tensor_info Tensor metadata
   * @return Constructed filename
   */
  std::string construct_output_filename(size_t iteration, size_t tensor_idx, const vart::NpuTensorInfo& tensor_info) {
    vart::DataType data_type = tensor_info.data_type;
    const std::vector<uint32_t>& tensor_shape = tensor_info.shape;
    const std::string& tensor_name = tensor_info.name;

    /* Convert tensor_shape to a string */
    std::ostringstream shape_stream;
    for (size_t k = 0; k < tensor_shape.size(); ++k) {
      shape_stream << tensor_shape[k];
      if (k < tensor_shape.size() - 1) {
        shape_stream << "x";
      }
    }
    std::string tensor_shape_str = shape_stream.str();

    /* Sanitize tensor name */
    std::string safe_name = tensor_name;
    replace(safe_name.begin(), safe_name.end(), '/', '-');
    if (!safe_name.empty() && safe_name.front() == '-') {
      safe_name.erase(0, 1);
    }

    /* Create filename */
    std::string append_str = dataTypeToString(data_type) + "_" + tensor_shape_str + "_";
    std::string infer_prefix = "infer_";
    std::string base_filename =
        infer_prefix + "out" + std::to_string(tensor_idx) + "-" + append_str + safe_name + ".bin";

    std::string file_name;
    if (m_app_opt.n_runs > 1) {
      /* Multiple iterations - include iteration number in filename */
      file_name = "iter" + std::to_string(iteration) + "_" + base_filename;
    } else {
      /* Single iteration */
      file_name = std::move(base_filename);
    }

    return file_name;
  }

  /**
   * @brief Open output files
   *
   * Opens one file per output tensor in truncate mode.
   *
   * @param iteration Current iteration number
   * @return Status of the operation
   */
  vart_app_status open_output_files(size_t iteration) {
    // Close any previously open files
    close_output_files();

    m_output_files.resize(m_num_output_tensors);
    m_output_file_paths.resize(m_num_output_tensors);

    for (size_t j = 0; j < m_num_output_tensors; ++j) {
      auto& tensor_info = m_output_tensors_info[j];

      /* Construct file name */
      std::string file_name = construct_output_filename(iteration, j, tensor_info);
      m_output_file_paths[j] = std::filesystem::path(m_app_opt.ofm_dir) / file_name;

      /* Open with truncate - ensures fresh file for each iteration */
      m_output_files[j].open(m_output_file_paths[j], std::ios::binary | std::ios::out | std::ios::trunc);

      if (!m_output_files[j].is_open()) {
        APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Unable to open file: %s", m_output_file_paths[j].c_str());
        close_output_files(); /* Clean up any opened files */
        return vart_app_status::FAILURE;
      }

      APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Opened output file for tensor %zu: %s", j,
              m_output_file_paths[j].c_str());
    }

    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Close all output files
   *
   * Closes all open output file handles and clears the file handle vectors.
   */
  void close_output_files() {
    for (size_t j = 0; j < m_output_files.size(); ++j) {
      if (m_output_files[j].is_open()) {
        m_output_files[j].close();
        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Closed output file for tensor %zu", j);
      }
    }
    m_output_files.clear();
    m_output_file_paths.clear();
  }

  /**
   * @brief Write output tensors
   *
   * Writes tensor data to already-open files.
   *
   * @param frame_count Current frame count to calculate offsets
   * @param actual_batch_size The actual number of frames processed
   * @return Status of the operation
   */
  vart_app_status write_output_tensors(size_t frame_count, size_t actual_batch_size) {
    try {
      /* Write each tensor's batch data */
      for (size_t j = 0; j < m_num_output_tensors; ++j) {
        auto& tensor_info = m_output_tensors_info[j];
        auto tensor_size_in_bytes = tensor_info.size_in_bytes;
        auto& file = m_output_files[j];

        /* Write all batch elements to the already-open file */
        for (size_t i = 0; i < actual_batch_size; ++i) {
          /* Calculate the offset based on absolute frame number */
          size_t absolute_frame_number = frame_count + i;
          size_t offset = absolute_frame_number * tensor_size_in_bytes;

          file.seekp(offset, std::ios::beg);
          if (!file) {
            APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Unable to seek to offset %zu in file %s", offset,
                    m_output_file_paths[j].c_str());
            return vart_app_status::FAILURE;
          }

          /* Source of the PL kernel output: in zero-copy mode the kernel writes
           * into a persistent PL output bo, so read straight from its host
           * mapping (no from-PL memcpy). Otherwise the host-copy path wrote the
           * result back over the NPU output tensor, so read that. */
          const void* data_ptr = (m_pl && m_pl_zerocopy)
                                     ? static_cast<const void*>(m_pl_out_ptrs[i][j])
                                     : m_outputs[i][j].get_virtual_address();
          if (!data_ptr) {
            APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to get virtual address for output tensor[%zu][%zu]",
                    i, j);
            return vart_app_status::FAILURE;
          }
          file.write(static_cast<const char*>(data_ptr), tensor_size_in_bytes);

          if (!file) {
            APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Unable to write data to file %s at offset %zu",
                    m_output_file_paths[j].c_str(), offset);
            return vart_app_status::FAILURE;
          }
        }

        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Wrote tensor %zu data for frames %zu-%zu to file: %s", j,
                frame_count, frame_count + actual_batch_size - 1, m_output_file_paths[j].c_str());
        std::cout << "Wrote tensor " << j << " data for frames " << frame_count << "-"
                  << (frame_count + actual_batch_size - 1) << " to file: " << m_output_file_paths[j] << std::endl;
      }
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error writing output tensors: %s", e.what());
      return vart_app_status::FAILURE;
    }

    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Run inference and optionally save output tensors.
   * @param benchmark Indicates if benchmark mode is enabled to measure performance metrics.
   * @param options Reference to the app_opt structure containing application configuration options.
   * @param iteration Current iteration number for tracking multiple inference runs.
   * @param total_frames Total number of frames to process during inference.
   * @param total_inference_time Reference to a double variable to accumulate total inference time in microseconds.
   * @return Returns 0 on success, 1 on failure.
   */
  int run_inference_and_save(bool benchmark,
                             const app_opt& options,
                             size_t iteration,
                             size_t total_frames,
                             double& total_inference_time) {
    size_t frame_count = 0;  // Initialize frame count
    size_t num_full_batches = 0;
    size_t num_partial_batches = 0;
    APP_LOG(AppLogLevel::DEBUG, options.app_log, "Total frames to process: %zu", total_frames);

    /* Open and truncate all output files at iteration start (unless dry_run or benchmark) */
    if (!options.dry_run && !benchmark) {
      if (vart_app_status::FAILURE == open_output_files(iteration)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to open output files for iteration %zu", iteration);
        return 1;
      }
    }

    while (frame_count < total_frames) {
      /* Calculate actual batch size for this iteration (handles partial batches at end) */
      size_t actual_batch_size = std::min(m_batch_size, total_frames - frame_count);
      bool is_partial_batch = (actual_batch_size < m_batch_size);

      if (is_partial_batch) {
        APP_LOG(AppLogLevel::INFO, options.app_log, "Processing partial batch: %zu frames (frame %zu-%zu of %zu)",
                actual_batch_size, frame_count, frame_count + actual_batch_size - 1, total_frames);
      }

      if (!m_app_opt.dry_run) {
        /* Populate input tensors for the current run */
        if (vart_app_status::FAILURE == populate_input_tensors(frame_count, actual_batch_size)) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to populate input tensors at frame %zu", frame_count);
          return 1;
        }
      }

      /* Output NpuTensors are already prepared in allocate_output_tensors() using Runner */
      /* Run inference */
      auto start = std::chrono::high_resolution_clock::now();  // Start timestamp for inference
      if (vart_app_status::FAILURE == run_inference()) {
        APP_LOG(AppLogLevel::DEBUG, options.app_log, "Failed to run inference");
        return 1;
      }
      auto end = std::chrono::high_resolution_clock::now();  // End timestamp for inference

      if (benchmark) {
        // Calculate duration for this run and accumulate
        double duration_us =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        total_inference_time += duration_us;
      }

      /* Track batch statistics */
      if (is_partial_batch) {
        num_partial_batches++;
      } else {
        num_full_batches++;
      }

      if (options.dry_run) {
        /* Dry run measures nothing past inference: skip the PL forward and the
         * output file writes entirely. */
        APP_LOG(AppLogLevel::DEBUG, options.app_log,
                "Completed batch: frames %zu-%zu (%zu frames), total processed: %zu/%zu", frame_count,
                frame_count + actual_batch_size - 1, actual_batch_size, frame_count + actual_batch_size, total_frames);

        frame_count += actual_batch_size;  // Increment frame count by actual batch size
        continue;
      }

      /* Forward the inference outputs through the pass_through PL kernel and
       * overwrite each output tensor in place with the kernel result, so the
       * data written to file is the PL kernel output. In benchmark mode this
       * is still run so its per-stage timings are measured; only the file
       * writes below are skipped. */
      if (vart_app_status::FAILURE == forward_outputs_through_pl(actual_batch_size)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to forward outputs through PL kernel for frame %zu",
                frame_count);
        if (!benchmark) {
          close_output_files();  // Clean up on error
        }
        return 1;
      }

      if (benchmark) {
        /* Skip saving output tensors in benchmark mode (files were not opened). */
        APP_LOG(AppLogLevel::DEBUG, options.app_log,
                "Completed batch: frames %zu-%zu (%zu frames), total processed: %zu/%zu", frame_count,
                frame_count + actual_batch_size - 1, actual_batch_size, frame_count + actual_batch_size, total_frames);

        frame_count += actual_batch_size;  // Increment frame count by actual batch size
        continue;
      }

      /* Write output tensors to files */
      if (vart_app_status::FAILURE == write_output_tensors(frame_count, actual_batch_size)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to write output tensors for frame %zu", frame_count);
        close_output_files();  // Clean up on error
        return 1;
      }

      /* Log batch completion before incrementing frame_count */
      APP_LOG(AppLogLevel::DEBUG, options.app_log,
              "Completed batch: frames %zu-%zu (%zu frames), total processed: %zu/%zu", frame_count,
              frame_count + actual_batch_size - 1, actual_batch_size, frame_count + actual_batch_size, total_frames);

      frame_count += actual_batch_size;  // Increment frame count by actual batch size
    }

    /* Log batch processing summary */
    size_t total_batches = num_full_batches + num_partial_batches;
    if (num_partial_batches > 0) {
      size_t partial_batch_frames = total_frames % m_batch_size;
      APP_LOG(AppLogLevel::RESULT, options.app_log,
              "Batch processing summary: %zu total frames, %zu batches (%zu full, %zu partial with %zu frames)",
              total_frames, total_batches, num_full_batches, num_partial_batches, partial_batch_frames);
    } else {
      APP_LOG(AppLogLevel::RESULT, options.app_log, "Batch processing summary: %zu total frames, %zu full batches",
              total_frames, num_full_batches);
    }

    /* Close all output files at iteration end */
    if (!options.dry_run && !benchmark) {
      close_output_files();
    }

    return 0;
  }

  /**
   * @brief Logs the average inference time based on the total frames, total inference time, and total runs.
   * @param total_frames The total number of frames processed during inference.
   * @param total_inference_time The accumulated inference time in microseconds across all runs.
   * @param total_runs The total number of inference runs performed.
   */
  void log_average_inference_time(size_t total_frames, double total_inference_time, size_t total_runs) {
    const double denom = static_cast<double>(total_frames * total_runs);
    double avg_time_us = total_inference_time / denom;
    double avg_time_ms = avg_time_us / 1000;
    std::ostringstream time_str;
    time_str << std::fixed << std::setprecision(2) << avg_time_ms;
    std::cout << "Average inference time over " << total_runs << " runs (ML only): " << time_str.str() << " ms" << std::endl;

    /* Per-stage breakdown of the ml_vart_plus_pl datapath, each averaged per
     * frame. The PL sub-stages come from the pass_through wrapper's internal
     * timers (accumulated across all runs):
     *   ML inference        : NPU run_inference() (== the average above)
     *   data-transfer-to-PL : host->PL input staging (0 for the zero-copy path)
     *   PL dummy post processing: pass_through kernel launch + wait
     *   data-transfer-from-PL: PL->host result copy-back (reported for context) */
    if (m_pl && m_pl_forward_frames > 0) {
      const double pl_denom = static_cast<double>(m_pl_forward_frames);
      const double ml_ms = avg_time_ms;
      const double to_pl_ms = (m_pl->to_pl_us() / pl_denom) / 1000.0;
      const double pl_exec_ms = (m_pl->pl_exec_us() / pl_denom) / 1000.0;
      const double from_pl_ms = (m_pl->from_pl_us() / pl_denom) / 1000.0;

      std::cout << std::fixed << std::setprecision(3);
      std::cout << "Per-stage average (ms/frame, " << (m_pl_zerocopy ? "zero-copy" : "host-copy")
                << " ML->PL):" << std::endl;
      std::cout << "  ML inference             : " << ml_ms << std::endl;
      std::cout << "  data-transfer-to-PL      : " << to_pl_ms << std::endl;
      std::cout << "  PL dummy post processing : " << pl_exec_ms << std::endl;
      std::cout << "  data-transfer-from-PL    : " << from_pl_ms << std::endl;

      /* End-to-end total = sum of the four pipeline stages above. This is the
       * full per-frame latency of the ml_vart_plus_pl datapath (NPU inference
       * plus the PL forward: input staging + pass_through kernel + result
       * copy-back), as opposed to the "ML only" headline number. */
      const double total_ms = ml_ms + to_pl_ms + pl_exec_ms + from_pl_ms;
      std::cout << "  ------------------------------------" << std::endl;
      std::cout << "  total (end-to-end)       : " << total_ms << std::endl;
      std::cout.unsetf(std::ios::floatfield);
    }
  }
};

/**
 * @brief Parse the JSON configuration file and populate model options.
 *
 * This function reads a JSON configuration file containing model settings, input/output
 * file paths, and runner options. It populates app_opt and runner_opt structures
 * configuration parameters needed for inference.
 *
 * Expected JSON structure:
 * {
 *   "inference-config": {
 *     "model-file": "<path-to-model-cache-dir>",
 *     "runner-options": {
 *       "log-level": "ERROR|WARNING|INFO|DEBUG",
 *       "config-file": "<optional-vitis-ai-config-json>",
 *       "aie-columns-sharing": true|false,
 *       "start-column": <optional-uint>,
 *       "cma-index": <optional-int>,
 *       "ai-analyzer-profiling": true|false
 *     }
 *   },
 *   "ifms-config": [
 *     { "name": "<runner-input-tensor-name-1>", "file": "<path-to-input-file-1.bin>" },
 *     { "name": "<runner-input-tensor-name-2>", "file": "<path-to-input-file-2.bin>" }
 *   ]
 * }
 * @param config_file Path to configuration JSON file
 * @param app_info Output application options structure to be populated
 * @param runner_info Output runner options structure to be populated
 * @throws std::exception if parsing fails or required keys are missing
 */
void parse_config_file(const std::string& config_file, app_opt& app_info, runner_opt& runner_info) {
  try {
    ptree config;
    boost::property_tree::read_json(config_file, config);

    runner_info.model_cache_dir = config.get<std::string>("inference-config.model-file");

    if (auto runner_opt = config.get_child_optional("inference-config.runner-options")) {
      const auto& runner_options = runner_opt.get();
      runner_info.log_level = runner_options.get<std::string>("log-level", runner_info.log_level);
      runner_info.config_json = runner_options.get<std::string>("config-file", runner_info.config_json);
      runner_info.aie_columns_sharing =
          runner_options.get<bool>("aie-columns-sharing", runner_info.aie_columns_sharing);
      if (auto sc = runner_options.get_optional<uint32_t>("start-column")) {
        runner_info.start_column = sc.get();
        runner_info.is_start_column_set = true;
      }
      if (auto ci = runner_options.get_optional<int32_t>("cma-index")) {
        runner_info.cma_index = ci.get();
        runner_info.is_cma_index_set = true;
      }
      runner_info.ai_analyzer_profiling =
          runner_options.get<bool>("ai-analyzer-profiling", runner_info.ai_analyzer_profiling);
      runner_info.input_tensor_type =
          runner_options.get<std::string>("input-tensor-type", runner_info.input_tensor_type);
      runner_info.output_tensor_type =
          runner_options.get<std::string>("output-tensor-type", runner_info.output_tensor_type);

      if (runner_info.input_tensor_type != "CPU" && runner_info.input_tensor_type != "HW") {
        APP_LOG(AppLogLevel::WARNING, app_info.app_log,
                "Invalid input-tensor-type '%s'. Supported values are CPU and HW. Falling back to HW.",
                runner_info.input_tensor_type.c_str());
        runner_info.input_tensor_type = "HW";
      }

      if (runner_info.output_tensor_type != "CPU" && runner_info.output_tensor_type != "HW") {
        APP_LOG(AppLogLevel::WARNING, app_info.app_log,
                "Invalid output-tensor-type '%s'. Supported values are CPU and HW. Falling back to HW.",
                runner_info.output_tensor_type.c_str());
        runner_info.output_tensor_type = "HW";
      }
    }

    if (auto ifms_config = config.get_child_optional("ifms-config")) {
      size_t idx = 0;
      for (const auto& ifm : ifms_config.get()) {
        const std::string name = ifm.second.get<std::string>("name", "");
        const std::string file = ifm.second.get<std::string>("file", "");
        if (name.empty()) {
          APP_LOG(AppLogLevel::ERROR, app_info.app_log,
                  "ifms-config[%zu]: 'name' is missing or empty; it must match the runner-reported input tensor name "
                  "(use 'ml_vart --get-model-info %s' to discover the input tensor names).",
                  idx, runner_info.model_cache_dir.c_str());
          throw std::runtime_error("Invalid ifms-config: missing 'name'");
        }
        if (file.empty()) {
          APP_LOG(AppLogLevel::ERROR, app_info.app_log, "ifms-config[%zu] (name='%s'): 'file' is missing or empty.",
                  idx, name.c_str());
          throw std::runtime_error("Invalid ifms-config: missing 'file'");
        }
        auto inserted = app_info.ifm_files_by_name.emplace(name, file);
        if (!inserted.second) {
          APP_LOG(AppLogLevel::ERROR, app_info.app_log,
                  "ifms-config[%zu]: duplicate 'name'='%s' (previously bound to file '%s').", idx, name.c_str(),
                  inserted.first->second.c_str());
          throw std::runtime_error("Invalid ifms-config: duplicate 'name'");
        }
        ++idx;
      }
    }

    if (!app_info.dry_run && app_info.ifm_files_by_name.empty()) {
      APP_LOG(AppLogLevel::ERROR, app_info.app_log, "Missing required JSON field 'ifms-config'.");
      throw std::runtime_error("Missing required field: ifms-config");
    }

    // Access "ofms-dir" at the root level
    app_info.ofm_dir = config.get<std::string>("ofms-dir", "output");

    // Check if the directory exists, and create it if it doesn't
    if (!app_info.dry_run && !std::filesystem::exists(app_info.ofm_dir)) {
      try {
        std::filesystem::create_directories(app_info.ofm_dir);
        APP_LOG(AppLogLevel::INFO, app_info.app_log, "Created output directory: %s", app_info.ofm_dir.c_str());
      } catch (const std::filesystem::filesystem_error& e) {
        APP_LOG(AppLogLevel::ERROR, app_info.app_log, "Failed to create output directory: %s", e.what());
        throw;
      }
    }

    /* pass_through PL kernel configuration: the .xclbin containing the
     * pass_through kernel and (optionally) the kernel name to instantiate. */
    if (auto pl_config = config.get_child_optional("pl-config")) {
      const auto& pl = pl_config.get();
      app_info.pl_xclbin = pl.get<std::string>("xclbin-location", app_info.pl_xclbin);
      app_info.pl_kernel = pl.get<std::string>("kernel-name", app_info.pl_kernel);
      app_info.pl_device_index = pl.get<unsigned int>("device-index", app_info.pl_device_index);
    }

    if (!app_info.dry_run && app_info.pl_xclbin.empty()) {
      APP_LOG(AppLogLevel::ERROR, app_info.app_log,
              "Missing required JSON field 'pl-config.xclbin-location' (path to the .xclbin containing the "
              "pass_through PL kernel).");
      throw std::runtime_error("Missing required field: pl-config.xclbin-location");
    }

    app_info.print();
    print_runner_opt(runner_info, app_info.app_log);

    /* Validate paths when dry-run if false */
    if (!app_info.dry_run && vart_app_status::FAILURE == app_info.validate_app_opt(runner_info)) {
      throw std::runtime_error("Path validation failed");
    }
  } catch (const boost::property_tree::json_parser_error& e) {
    APP_LOG(AppLogLevel::ERROR, app_info.app_log, "Error reading JSON config: %s", e.what());
    throw;
  } catch (const boost::property_tree::ptree_bad_path& e) {
    APP_LOG(AppLogLevel::ERROR, app_info.app_log, "Config missing required key: %s", e.what());
    throw;
  } catch (const std::exception& e) {
    APP_LOG(AppLogLevel::ERROR, app_info.app_log, "Error parsing config: %s", e.what());
    throw;
  }
}

/**
 * @brief Parse model runner arguments and initialize the application context.
 *
 * This function serves as a wrapper that:
 * 1. Validates that the configuration file exists
 * 2. Calls parse_config_file() to read and parse the JSON configuration
 * 3. Creates a new app_context instance with the parsed configuration
 *
 * The function checks for the existence of the config file specified in opt.app_config_file
 * before attempting to parse it. If the file is found and successfully parsed, a new
 * app_context object is created and assigned to the runner shared pointer.
 *
 * @param runner A shared pointer reference to app_context where the created
 *               application context will be stored. Will be nullptr on failure.
 * @param app_options A reference to app options containing command-line arguments.
 * @param runner_options A reference to runner options populated from JSON.
 * @return vart_app_status::SUCCESS if the config file was found, parsed, and context created;
 *         vart_app_status::FAILURE if the config file doesn't exist or parsing fails.
 */
vart_app_status parse_model_runner_args(std::shared_ptr<app_context>& runner,
                                        app_opt& app_options,
                                        runner_opt& runner_options) {
  try {
    const std::string config_file = app_options.app_config_file;
    if (std::ifstream(config_file)) {
      APP_LOG(AppLogLevel::INFO, app_options.app_log, "Config file found: %s. Parsing model configuration...",
              config_file.c_str());
      parse_config_file(config_file, app_options, runner_options);
      runner = std::make_shared<app_context>(app_options, runner_options);
    } else {
      APP_LOG(AppLogLevel::ERROR, app_options.app_log, "Config file not found: %s", config_file.c_str());
      return vart_app_status::FAILURE;
    }
  } catch (const std::exception& e) {
    return vart_app_status::FAILURE;
  }

  return vart_app_status::SUCCESS;
}

/**
 * @brief Parse command-line arguments and populate configuration options.
 *
 * This function processes all command-line arguments using getopt_long and
 * populates the app_opt structure with command-line values.
 * of numeric arguments and sets appropriate flags.
 *
 * Supported command-line options:
 * - --app-config <file>  : JSON configuration file path (mandatory)
 * - --runs <N>           : Number of inference iterations (default: 1)
 * - --frames <N>         : Number of frames to process (default: -1, all frames)
 * - --benchmark          : Enable benchmark mode (no output saving)
 * - --dry-run            : Skip file I/O operations
 * - --log-level <N>      : Application log level (1-6)
 * - --help               : Display help text
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @param options Reference to app_opt structure to populate with parsed values
 * @param benchmark Reference to boolean flag set if benchmark mode is enabled
 * @return 0 on success, -1 if help is requested or invalid options, 1 on parsing errors
 */
int read_user_inputs(int argc, char* argv[], app_opt& options, bool& benchmark) {
  static struct option long_options[] = {{"help", no_argument, nullptr, 0},
                                         {"app-config", required_argument, nullptr, 1},
                                         {"benchmark", no_argument, nullptr, 2},
                                         {"runs", required_argument, nullptr, 3},
                                         {"frames", required_argument, nullptr, 4},
                                         {"dry-run", no_argument, nullptr, 5},
                                         {"log-level", required_argument, nullptr, 6},
                                         {"get-model-info", required_argument, nullptr, 7},
                                         {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int option = 0;

  while ((option = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (option) {
      case 1:  // --app-config
        options.app_config_file = optarg;
        break;
      case 2:  // --benchmark
        benchmark = true;
        break;
      case 3:  // --runs
        try {
          unsigned long temp = std::stoul(optarg);
          if (temp == 0 || temp > std::numeric_limits<uint32_t>::max()) {
            APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --runs value is out of range: %s", optarg);
            return 1;
          }
          options.n_runs = static_cast<uint32_t>(temp);
        } catch (const std::invalid_argument&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: Invalid number for --runs: %s", optarg);
          return 1;
        } catch (const std::out_of_range&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --runs value is out of range: %s", optarg);
          return 1;
        }
        break;
      case 4:  // --frames
        try {
          long temp = std::stol(optarg);
          if (temp <= 0) {
            APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --frames must be greater than 0");
            return 1;
          }
          if (temp > std::numeric_limits<int>::max()) {
            APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --frames value is out of range: %s", optarg);
            return 1;
          }
          options.frames = static_cast<int>(temp);
        } catch (const std::invalid_argument&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: Invalid number for --frames: %s", optarg);
          return 1;
        } catch (const std::out_of_range&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --frames value is out of range: %s", optarg);
          return 1;
        }
        break;
      case 5:  // --dry-run
        options.dry_run = true;
        break;
      case 6:  // --log-level
        try {
          int log_level = std::stoi(optarg);
          if (log_level < static_cast<int>(AppLogLevel::ERROR) || log_level > static_cast<int>(AppLogLevel::DEBUG)) {
            APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --log-level must be between %d and %d",
                    static_cast<int>(AppLogLevel::ERROR), static_cast<int>(AppLogLevel::DEBUG));
            return 1;
          }
          options.app_log = static_cast<AppLogLevel>(log_level);
        } catch (const std::invalid_argument&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: Invalid number for --log-level: %s", optarg);
          return 1;
        } catch (const std::out_of_range&) {
          APP_LOG(AppLogLevel::ERROR, options.app_log, "Error: --log-level value is out of range: %s", optarg);
          return 1;
        }
        break;
      case 7:  // --get-model-info <model-path>
        options.get_model_info = true;
        options.get_model_info_path = optarg;
        break;
      case '?':  // '?' - Unknown option (getopt_long convention)
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Unknown option: %s", argv[optind - 1]);
        print_help_text(argv[0]);
        return -1;
      case 0:  // --help
      default:
        print_help_text(argv[0]);
        return -1;
    }
  }

  if (options.dry_run) {
    APP_LOG(AppLogLevel::INFO, options.app_log, "Dry run mode enabled. IFMs reading and OFMs saving will be skipped.");
  }

  return 0;
}

/**
 * @brief Main entry point
 * @return 0 on success, 1 on failure, -1 if help requested
 */
int main(int argc, char* argv[]) {
  app_opt options;
  runner_opt runner_options;
  bool benchmark = false;
  double total_inference_time = 0.0;
  size_t ifms_total_frames = 0;
  size_t total_frames = 1;

  try {
    int result = read_user_inputs(argc, argv, options, benchmark);
    if (result != 0) {
      return result;
    }
    /* `--app-config` is mandatory for normal/dry-run/benchmark inference flows
     * but is OPTIONAL when `--get-model-info <model-path>` is supplied: the
     * operator can inspect a model without authoring an app-config JSON. */
    /* `--app-config` is mandatory for normal/dry-run/benchmark inference flows
     * and is IGNORED when `--get-model-info <model-path>` is supplied: the
     * operator can inspect a model without authoring an app-config JSON. If
     * both are passed, --get-model-info wins and --app-config is dropped with
     * a WARNING (see below). */
    if (options.app_config_file.empty() && !options.get_model_info) {
      APP_LOG(AppLogLevel::ERROR, options.app_log,
              "--app-config is mandatory (or pass --get-model-info <model-path> to inspect tensor metadata only).");
      print_help_text(argv[0]);
      return -1;
    }

    /* When --get-model-info is supplied, the supplied path is what the runner
     * will load. Validate it up front so the operator gets a clear error
     * instead of a deeper VART failure. */
    if (options.get_model_info) {
      std::error_code ec;
      if (!std::filesystem::exists(options.get_model_info_path, ec) ||
          !(std::filesystem::is_regular_file(options.get_model_info_path, ec) ||
            std::filesystem::is_directory(options.get_model_info_path, ec))) {
        APP_LOG(AppLogLevel::ERROR, options.app_log,
                "--get-model-info path does not exist or is not a file/directory: %s",
                options.get_model_info_path.c_str());
        return -1;
      }
    }

    std::shared_ptr<app_context> runner = nullptr;

    if (options.get_model_info) {
      /* Standalone mode: --get-model-info ignores --app-config entirely.
       * Build a minimal runner-options struct from defaults + the supplied
       * model path, and force CPU input/output tensor type (some models with
       * CPU subgraph boundaries cannot be loaded with HW tensor type). */
      if (!options.app_config_file.empty()) {
        APP_LOG(AppLogLevel::WARNING, options.app_log, "--get-model-info is set; ignoring --app-config '%s'.",
                options.app_config_file.c_str());
      }
      runner_options = runner_opt{};
      runner_options.model_cache_dir = options.get_model_info_path;
      runner_options.input_tensor_type = "CPU";
      runner_options.output_tensor_type = "CPU";
      APP_LOG(AppLogLevel::INFO, options.app_log,
              "--get-model-info flag supplied: model info will be retrieved with CPU tensors.");
      runner = std::make_shared<app_context>(options, runner_options);
    } else {
      /* Normal/dry-run/benchmark inference flow: build runner from app-config. */
      if (vart_app_status::FAILURE == parse_model_runner_args(runner, options, runner_options)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to parse model runner arguments");
        return 1;
      }
    }

    /* Create VART runner */
    if (vart_app_status::FAILURE == runner->create_runner()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to create VART runner");
      return 1;
    }

    /* --get-model-info: dump the runner-reported tensor metadata to
     * console + <model_basename>_info.json and exit. No tensor allocation,
     * no inference, no IFM/OFM I/O. */
    if (options.get_model_info) {
      std::filesystem::path dump_path;
      if (vart_app_status::FAILURE == runner->dump_model_info_to_json(dump_path)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to dump model info JSON.");
        return 1;
      }
      return 0;
    }

    /* Inference / dry-run / benchmark flow: load (and at INFO+ log levels
     * print) the runner-reported tensor metadata, then bind user-supplied
     * ifms-config entries to runner-reported input tensors by name. Dry-run
     * skips the ifms binding because no IFM files are supplied. */
    if (vart_app_status::FAILURE == runner->get_tensor_metadata()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to get tensor metadata");
      return 1;
    }
    if (!options.dry_run) {
      if (vart_app_status::FAILURE == runner->resolve_ifms_in_runner_order()) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to resolve IFM files in runner-reported tensor order");
        return 1;
      }
    }

    if (!options.dry_run) {
      ifms_total_frames = runner->calculate_ifms_total_frames();  // Calculate total number of frames in the IFM
      if (ifms_total_frames == 0) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "No input frames available from IFM files");
        APP_LOG(AppLogLevel::INFO, options.app_log, "Exiting Application.");
        return 1;
      }
      total_frames = (options.frames > 0 && static_cast<size_t>(options.frames) < ifms_total_frames)
                         ? static_cast<size_t>(options.frames)
                         : ifms_total_frames;  // Determine total frames to process
    } else {
      total_frames = options.frames > 0 ? static_cast<size_t>(options.frames) : 1;
    }

    /* Allocate input tensors */
    if (runner->allocate_input_tensors()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to allocate input tensors before populating.");
      return -1;
    }

    /* Allocate output tensors */
    if (runner->allocate_output_tensors()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to allocate output tensors before populating.");
      return -1;
    }

    /* Initialize the pass_through PL kernel used to post-process the outputs */
    if (vart_app_status::FAILURE == runner->init_pl_kernel()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to initialize the pass_through PL kernel.");
      return -1;
    }

    /* Run the inference for n number of runs default is 1 */
    APP_LOG(AppLogLevel::DEBUG, options.app_log, "Running the inference for %u runs", options.n_runs);
    const size_t total_runs = static_cast<size_t>(options.n_runs);

    for (size_t r = 0; r < total_runs; ++r) {
      if (runner->run_inference_and_save(benchmark, options, r, total_frames, total_inference_time)) {
        APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to run inference and save output tensors for run %zu", r);
        return 1;
      }
    }

    /* End the timer and calculate average time */
    if (benchmark) {
      runner->log_average_inference_time(total_frames, total_inference_time, total_runs);
    }

    cout << "Run completed successfully." << std::endl;
    return 0;
  } catch (const std::exception& e) {
    APP_LOG(AppLogLevel::ERROR, options.app_log, "Unhandled exception: %s", e.what());
    APP_LOG(AppLogLevel::ERROR, options.app_log, "Exiting Application.");
    return 1;
  } catch (...) {
    APP_LOG(AppLogLevel::ERROR, options.app_log, "Unhandled exception: unknown error");
    APP_LOG(AppLogLevel::ERROR, options.app_log, "Exiting Application.");
    return 1;
  }
}
