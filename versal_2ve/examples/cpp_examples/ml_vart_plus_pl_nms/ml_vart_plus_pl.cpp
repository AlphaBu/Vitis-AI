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
#include <algorithm>
#include <chrono>
#include <cstdint>
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

/* Native XRT API used to drive the nms_onnx PL (Programmable Logic)
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

  /* nms_onnx PL kernel configuration (from JSON "pl-config").
   * - pl_xclbin: path to the .xclbin that contains the PL kernel. Mandatory
   *   for the inference flow because every inference output is post-processed
   *   through the NMS kernel and the kernel output becomes the app output.
   * - pl_kernel: kernel top-function name (default "nms_onnx").
   * - pl_device_index: XRT device index that hosts the PL kernel
   *   (default 1; on VEK385 the PL region is device 1, the NPU is 0). */
  std::string pl_xclbin;
  std::string pl_kernel;
  unsigned int pl_device_index;

  /* NMS scalar parameters (from JSON "pl-config"), matching the ONNX
   * NonMaxSuppression operator semantics:
   * - nms_num_classes: number of score classes in the model output.
   * - nms_max_out_per_class: cap on selections per class (0 = unlimited).
   * - nms_iou_threshold: IoU suppression threshold.
   * - nms_score_threshold: score filter threshold.
   * - nms_center_point_box: 1 = boxes are [cx,cy,w,h] (YOLOX), 0 = corners. */
  int nms_num_classes;
  int nms_max_out_per_class;
  float nms_iou_threshold;
  float nms_score_threshold;
  int nms_center_point_box;

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
    pl_kernel = "nms_onnx";
    pl_device_index = 1;
    nms_num_classes = 2;
    nms_max_out_per_class = 200;
    nms_iou_threshold = 0.65f;
    nms_score_threshold = 0.01f;
    nms_center_point_box = 1;
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
    APP_LOG(AppLogLevel::INFO, app_log, "nms_num_classes: %d", nms_num_classes);
    APP_LOG(AppLogLevel::INFO, app_log, "nms_max_out_per_class: %d", nms_max_out_per_class);
    APP_LOG(AppLogLevel::INFO, app_log, "nms_iou_threshold: %.4f", nms_iou_threshold);
    APP_LOG(AppLogLevel::INFO, app_log, "nms_score_threshold: %.4f", nms_score_threshold);
    APP_LOG(AppLogLevel::INFO, app_log, "nms_center_point_box: %d", nms_center_point_box);
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

/* ---------------------------------------------------------------------------
 * bf16 <-> float helpers
 *
 * bf16 (bfloat16) is simply the high 16 bits of an IEEE-754 binary32 (float):
 * sign(1) + exponent(8) + mantissa(7). The NMS kernel's bf16_t port type is
 * ap_float<16,8>, which has the identical bit layout. Converting is therefore a
 * pure bit operation, with round-to-nearest-even applied on the float->bf16
 * narrowing.
 * ------------------------------------------------------------------------- */

/* Widen a bf16 bit pattern to float by placing it in the high 16 bits. */
static inline float bf16_bits_to_float(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

/* Narrow a float to a bf16 bit pattern with round-to-nearest-even. */
static inline uint16_t float_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  /* Round-to-nearest-even: add 0x7FFF plus the lsb of the retained mantissa. */
  const uint32_t rounding_bias = 0x00007FFFu + ((u >> 16) & 1u);
  u += rounding_bias;
  return static_cast<uint16_t>(u >> 16);
}

/*
 * @class pl_nms
 * @brief Thin native-XRT wrapper around the `nms_onnx` PL (Programmable Logic)
 *        kernel (ONNX NonMaxSuppression).
 *
 * The nms_onnx HLS kernel has the signature:
 *     void nms_onnx(const bf16_t *boxes_in,      // gmem0, arg0
 *                   const bf16_t *scores_in,     // gmem1, arg1
 *                   int32_t      *selected_out,  // gmem2, arg2
 *                   int  num_batches,            // arg3
 *                   int  num_classes,            // arg4
 *                   int  num_boxes,              // arg5
 *                   int  max_out_per_class,      // arg6
 *                   bf16_t iou_threshold,        // arg7 (16-bit s_axilite)
 *                   bf16_t score_threshold,      // arg8 (16-bit s_axilite)
 *                   int  center_point_box,       // arg9
 *                   int32_t *num_selected);      // gmem3, arg10
 *
 * Input layouts (bf16 = 16-bit):
 *   boxes_in [num_batches * num_boxes * 4]   : per box [cx,cy,w,h] (center_point_box=1)
 *   scores_in[num_batches * num_classes * num_boxes] : CLASS-MAJOR
 * Output layouts (int32):
 *   selected_out[num_selected*3]   : triplets [batch_idx, class_idx, box_idx]
 *   num_selected[0]                : number of valid triplets
 *
 * configure() allocates four persistent device buffers (boxes/scores inputs,
 * selected/num_selected outputs) and maps them to host pointers. The caller
 * fills boxes_host()/scores_host() (see the host-side model-output conversion),
 * then run() syncs inputs to the device, launches the kernel, and syncs the
 * results back.
 */
class pl_nms {
 public:
  pl_nms(const std::string& xclbin_path,
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
   * @brief Allocate the persistent NMS input/output device buffers and cache
   *        the scalar kernel arguments. Call once after the output dimensions
   *        are known (see app_context::init_pl_kernel).
   *
   * @param num_batches       NMS batch count (1 for per-frame invocation).
   * @param num_classes       Number of score classes.
   * @param num_boxes         Number of candidate boxes (anchors) per batch.
   * @param max_out_per_class Cap on selections per class (0 = unlimited).
   * @param iou_threshold     IoU suppression threshold (float, stored as bf16).
   * @param score_threshold   Score filter threshold (float, stored as bf16).
   * @param center_point_box  0 = corner boxes, 1 = center boxes [cx,cy,w,h].
   */
  void configure(int num_batches, int num_classes, int num_boxes,
                 int max_out_per_class, float iou_threshold,
                 float score_threshold, int center_point_box) {
    m_num_batches = num_batches;
    m_num_classes = num_classes;
    m_num_boxes = num_boxes;
    m_max_out_per_class = max_out_per_class;
    m_center_point_box = center_point_box;
    m_iou_bits = float_to_bf16_bits(iou_threshold);
    m_score_bits = float_to_bf16_bits(score_threshold);

    const size_t n_boxes_elems = static_cast<size_t>(num_batches) * num_boxes * 4;
    const size_t n_scores_elems = static_cast<size_t>(num_batches) * num_classes * num_boxes;
    m_selected_capacity = static_cast<size_t>(num_batches) * num_classes * num_boxes;

    const size_t boxes_bytes = n_boxes_elems * sizeof(uint16_t);
    const size_t scores_bytes = n_scores_elems * sizeof(uint16_t);
    const size_t selected_bytes = m_selected_capacity * 3 * sizeof(int32_t);

    m_boxes_bo = xrt::bo(m_device, boxes_bytes, m_kernel.group_id(0));      /* gmem0 */
    m_scores_bo = xrt::bo(m_device, scores_bytes, m_kernel.group_id(1));    /* gmem1 */
    m_selected_bo = xrt::bo(m_device, selected_bytes, m_kernel.group_id(2)); /* gmem2 */
    m_numsel_bo = xrt::bo(m_device, sizeof(int32_t), m_kernel.group_id(10)); /* gmem3 */

    m_boxes_host = m_boxes_bo.map<uint16_t*>();
    m_scores_host = m_scores_bo.map<uint16_t*>();
    m_selected_host = m_selected_bo.map<int32_t*>();
    m_numsel_host = m_numsel_bo.map<int32_t*>();

    APP_LOG(AppLogLevel::INFO, m_app_log,
            "NMS configured: batches=%d classes=%d boxes=%d max_out=%d "
            "iou=%.4f score=%.4f center=%d (boxes=%zuB scores=%zuB selected=%zuB)",
            num_batches, num_classes, num_boxes, max_out_per_class, iou_threshold,
            score_threshold, center_point_box, boxes_bytes, scores_bytes, selected_bytes);
  }

  /* Host mappings the caller fills before run(): boxes (bf16 [b*num_boxes*4],
   * layout [cx,cy,w,h]) and scores (bf16 [b*num_classes*num_boxes], class-major). */
  uint16_t* boxes_host() { return m_boxes_host; }
  uint16_t* scores_host() { return m_scores_host; }

  /**
   * @brief Sync the converted inputs to the device, launch the NMS kernel, wait
   *        for completion, and sync the results back. Timers separate the
   *        host->PL sync (to-PL), the kernel execution (PL), and the PL->host
   *        sync (from-PL).
   */
  void run() {
    /* Stage 1 (to-PL): push the converted inputs to the device. */
    const auto t0 = std::chrono::high_resolution_clock::now();
    m_boxes_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    m_scores_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    /* Stage 2 (PL exec): launch the kernel and wait. Argument order matches the
     * kernel signature. bf16 thresholds are passed as their 16-bit patterns. */
    const auto t1 = std::chrono::high_resolution_clock::now();
    auto run = m_kernel(m_boxes_bo, m_scores_bo, m_selected_bo, m_num_batches,
                        m_num_classes, m_num_boxes, m_max_out_per_class, m_iou_bits,
                        m_score_bits, m_center_point_box, m_numsel_bo);
    run.wait();

    /* Stage 3 (from-PL): sync the count and the selected triplets back. */
    const auto t2 = std::chrono::high_resolution_clock::now();
    m_numsel_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    m_selected_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto t3 = std::chrono::high_resolution_clock::now();

    m_to_pl_us += us(t0, t1);
    m_pl_exec_us += us(t1, t2);
    m_from_pl_us += us(t2, t3);
  }

  /* Results of the last run(): the number of selected triplets and a pointer to
   * the selected_out buffer (num_selected()*3 int32 values [batch,class,box]). */
  int num_selected() const { return m_numsel_host[0]; }
  const int32_t* selected() const { return m_selected_host; }
  size_t selected_capacity() const { return m_selected_capacity; }

  /* Accumulated sub-stage timers (microseconds), summed over every run() call
   * since the last reset_timers():
   *   to_pl_us   : host->PL input sync (boxes + scores TO_DEVICE)
   *   pl_exec_us : nms_onnx kernel launch + wait (the NMS PL kernel)
   *   from_pl_us : PL->host output sync (num_selected + selected FROM_DEVICE) */
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

  xrt::device m_device;
  xrt::kernel m_kernel;
  AppLogLevel m_app_log;

  /* Persistent NMS device buffers and their host mappings. */
  xrt::bo m_boxes_bo;
  xrt::bo m_scores_bo;
  xrt::bo m_selected_bo;
  xrt::bo m_numsel_bo;
  uint16_t* m_boxes_host = nullptr;
  uint16_t* m_scores_host = nullptr;
  int32_t* m_selected_host = nullptr;
  int32_t* m_numsel_host = nullptr;

  /* Cached scalar kernel arguments. */
  int m_num_batches = 1;
  int m_num_classes = 0;
  int m_num_boxes = 0;
  int m_max_out_per_class = 0;
  int m_center_point_box = 1;
  uint16_t m_iou_bits = 0;
  uint16_t m_score_bits = 0;
  size_t m_selected_capacity = 0;

  /* Per-stage timing accumulators (see the getters above). */
  double m_to_pl_us = 0.0;
  double m_pl_exec_us = 0.0;
  double m_from_pl_us = 0.0;
}; /* pl_nms */

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

  /* NMS output file handle. The NMS results are variable length per frame, so
   * they are written sequentially (frame order) to a single file rather than
   * one fixed-layout file per output tensor. */
  std::fstream m_nms_out_file;
  std::filesystem::path m_nms_out_path;

  /* vart runner */
  std::shared_ptr<vart::Runner> m_runner = nullptr;

  /* nms_onnx PL kernel wrapper (native XRT). Created in init_pl_kernel()
   * for the inference flow; stays null in dry-run / get-model-info. */
  std::unique_ptr<pl_nms> m_pl = nullptr;

  /* NMS dimensions derived from the model output tensor in init_pl_kernel(). */
  int m_nms_num_boxes = 0;         /* candidate boxes (anchors) per frame        */
  int m_nms_num_classes = 0;       /* score classes                              */
  int m_nms_stride = 0;            /* elements per anchor in the model output    */
  int m_nms_max_out_per_class = 0; /* cap on selections per class (0 = unlimited) */

  /* Per-frame NMS results captured in forward_outputs_through_pl():
   *   m_nms_counts[i]  = number of selected triplets for frame i
   *   m_nms_results[i] = flattened selected triplets [batch,class,box] * count */
  std::vector<int32_t> m_nms_counts;
  std::vector<std::vector<int32_t>> m_nms_results;

  /* Accumulated host-side model-output -> NMS-input conversion time
   * (microseconds), summed over every frame; reported separately in the
   * per-stage benchmark. The PL sub-stage timers live inside m_pl. */
  double m_conv_us = 0.0;

  /* Number of frames the PL forward has covered (summed across all runs), used
   * as the denominator for the per-stage per-frame timing report. */
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
   * @brief Create the nms_onnx PL kernel wrapper and configure it from the
   *        model output geometry and app options.
   *
   * Mandatory for the inference flow: this application converts every
   * inference output tensor into the NMS kernel's boxes/scores layout, runs
   * the NMS PL kernel, and writes the selected detections as the application
   * output. Skipped in dry-run (no real data to forward).
   *
   * The NMS dimensions are derived from the (single) model output tensor:
   * its last dimension is the per-anchor stride ([cx,cy,w,h,obj,cls0..] for
   * YOLOX), and num_boxes = (total elements) / stride. num_classes,
   * max_out_per_class and the thresholds come from the app options.
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
    if (m_output_tensors_info.empty()) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "No model output tensor available to drive NMS.");
      return vart_app_status::FAILURE;
    }
    /* Derive NMS geometry from the model output tensor. The kernel expects
     * one detection stream: boxes [num_boxes*4] and scores [num_classes*
     * num_boxes]. The model output is [.., num_boxes, stride] where stride =
     * 4 box coords + 1 objectness + num_classes class scores (+ padding). */
    const vart::NpuTensorInfo& oi = m_output_tensors_info[0];
    size_t total = 1;
    for (uint32_t d : oi.shape) total *= d;
    m_nms_stride = oi.shape.empty() ? 0 : static_cast<int>(oi.shape.back());
    if (m_nms_stride <= 0 || total % static_cast<size_t>(m_nms_stride) != 0) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
              "Unexpected model output geometry for NMS (elements=%zu stride=%d).", total, m_nms_stride);
      return vart_app_status::FAILURE;
    }
    m_nms_num_boxes = static_cast<int>(total / static_cast<size_t>(m_nms_stride));
    m_nms_num_classes = m_app_opt.nms_num_classes;
    m_nms_max_out_per_class = m_app_opt.nms_max_out_per_class;
    /* The per-anchor stride must hold 4 box coords + 1 objectness +
     * num_classes class scores; anything beyond that is padding. */
    if (m_nms_stride < 5 + m_nms_num_classes) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
              "Model output stride %d too small for 4 box coords + obj + %d classes.", m_nms_stride,
              m_nms_num_classes);
      return vart_app_status::FAILURE;
    }
    try {
      m_pl = std::make_unique<pl_nms>(m_app_opt.pl_xclbin, m_app_opt.pl_kernel, m_app_opt.app_log,
                                      m_app_opt.pl_device_index);
      m_pl->configure(/*num_batches=*/1, m_nms_num_classes, m_nms_num_boxes, m_nms_max_out_per_class,
                      m_app_opt.nms_iou_threshold, m_app_opt.nms_score_threshold,
                      m_app_opt.nms_center_point_box);
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Failed to initialize NMS PL kernel: %s", e.what());
      return vart_app_status::FAILURE;
    }
    /* Per-frame NMS result storage (count + [batch,class,box] triplets). */
    m_nms_counts.assign(m_batch_size, 0);
    m_nms_results.assign(m_batch_size, {});
    APP_LOG(AppLogLevel::RESULT, m_app_opt.app_log,
            "NMS configured: num_boxes=%d stride=%d num_classes=%d max_out_per_class=%d", m_nms_num_boxes,
            m_nms_stride, m_nms_num_classes, m_nms_max_out_per_class);
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Convert one frame's model output tensor into the NMS kernel's
   *        boxes/scores layout.
   *
   * This is the standalone data-conversion step that sits between the model
   * output and NMS processing (timed separately in --benchmark). The model
   * output for frame `i` is [num_boxes, stride] bf16 where each anchor is
   * [cx, cy, w, h, obj, cls0, cls1, ...(padding)]. The NMS kernel expects two
   * separate bf16 arrays:
   *   - boxes  [num_boxes*4]           : the 4 box coords per anchor (copied
   *                                       verbatim as bf16 bit patterns).
   *   - scores [num_classes*num_boxes] : class-major confidence = obj * cls_c
   *                                       (YOLOX combines objectness and class
   *                                       probability; neither the NPU model
   *                                       nor the NMS kernel does this multiply,
   *                                       so it is done here).
   *
   * @param i Frame index within the current batch (must be < actual_batch_size).
   * @return true on success, false if the output virtual address is null.
   */
  bool convert_model_output_to_nms_input(size_t i) {
    const uint16_t* mo = static_cast<const uint16_t*>(m_outputs[i][0].get_virtual_address());
    if (!mo) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log,
              "Failed to get virtual address for output tensor[%zu][0]", i);
      return false;
    }
    uint16_t* boxes = m_pl->boxes_host();
    uint16_t* scores = m_pl->scores_host();
    const int stride = m_nms_stride;
    const int num_boxes = m_nms_num_boxes;
    const int num_classes = m_nms_num_classes;
    for (int n = 0; n < num_boxes; ++n) {
      const uint16_t* a = mo + static_cast<size_t>(n) * stride;
      /* Box coords [cx,cy,w,h]: pure bf16 bit-pattern copy. */
      boxes[static_cast<size_t>(n) * 4 + 0] = a[0];
      boxes[static_cast<size_t>(n) * 4 + 1] = a[1];
      boxes[static_cast<size_t>(n) * 4 + 2] = a[2];
      boxes[static_cast<size_t>(n) * 4 + 3] = a[3];
      /* Confidence = objectness * class probability (class-major layout). */
      const float obj = bf16_bits_to_float(a[4]);
      for (int c = 0; c < num_classes; ++c) {
        const float conf = obj * bf16_bits_to_float(a[5 + c]);
        scores[static_cast<size_t>(c) * num_boxes + n] = float_to_bf16_bits(conf);
      }
    }
    return true;
  }

  /**
   * @brief Convert this batch's inference outputs and run them through the NMS
   *        PL kernel.
   *
   * For each valid frame: (1) the model output is converted into the kernel's
   * boxes/scores layout by convert_model_output_to_nms_input() (host-side,
   * timed separately into m_conv_us in --benchmark), then (2) the NMS kernel
   * is launched (its own to-PL / exec / from-PL stages are timed inside the
   * pl_nms wrapper). The selected detections (count + [batch,class,box]
   * triplets) are copied into m_nms_counts / m_nms_results for later output.
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
        /* Stage 1: host-side data conversion (model output -> NMS layout).
         * Timed separately so --benchmark can report it on its own line. */
        const auto conv_t0 = std::chrono::high_resolution_clock::now();
        if (!convert_model_output_to_nms_input(i)) {
          return vart_app_status::FAILURE;
        }
        const auto conv_t1 = std::chrono::high_resolution_clock::now();
        m_conv_us += std::chrono::duration<double, std::micro>(conv_t1 - conv_t0).count();

        /* Stage 2: run the NMS PL kernel (to-PL / exec / from-PL timed inside). */
        m_pl->run();

        /* Capture the selected detections for this frame. */
        const int count = m_pl->num_selected();
        const int32_t* sel = m_pl->selected();
        const size_t cap = m_pl->selected_capacity();
        const size_t take = std::min(static_cast<size_t>(std::max(count, 0)), cap);
        m_nms_counts[i] = static_cast<int32_t>(take);
        m_nms_results[i].assign(sel, sel + take * 3);

        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log,
                "Frame[%zu]: NMS selected %d detection(s)", i, static_cast<int>(take));
      }
      m_pl_forward_frames += actual_batch_size;
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error running NMS PL kernel: %s", e.what());
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
   * @brief Open the NMS results output file.
   *
   * A single binary file holds the selected detections for every frame,
   * written sequentially (the per-frame result is variable-length so fixed
   * offsets cannot be used). Opened in truncate mode at iteration start.
   *
   * @param iteration Current iteration number
   * @return Status of the operation
   */
  vart_app_status open_output_files(size_t iteration) {
    close_output_files();

    std::string file_name = "nms_selected.bin";
    if (m_app_opt.n_runs > 1) {
      file_name = "iter" + std::to_string(iteration) + "_" + file_name;
    }
    m_nms_out_path = std::filesystem::path(m_app_opt.ofm_dir) / file_name;

    m_nms_out_file.open(m_nms_out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!m_nms_out_file.is_open()) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Unable to open file: %s", m_nms_out_path.c_str());
      return vart_app_status::FAILURE;
    }
    APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Opened NMS results file: %s", m_nms_out_path.c_str());
    return vart_app_status::SUCCESS;
  }

  /**
   * @brief Close the NMS results output file.
   */
  void close_output_files() {
    if (m_nms_out_file.is_open()) {
      m_nms_out_file.close();
      APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Closed NMS results file");
    }
  }

  /**
   * @brief Write this batch's NMS results to the already-open results file.
   *
   * Each frame is written sequentially as:
   *   int32 num_selected
   *   num_selected * 3 * int32  (triplets [batch_idx, class_idx, box_idx])
   *
   * @param frame_count Current absolute frame count (for logging only).
   * @param actual_batch_size The actual number of frames processed.
   * @return Status of the operation
   */
  vart_app_status write_output_tensors(size_t frame_count, size_t actual_batch_size) {
    try {
      for (size_t i = 0; i < actual_batch_size; ++i) {
        const int32_t count = m_nms_counts[i];
        m_nms_out_file.write(reinterpret_cast<const char*>(&count), sizeof(int32_t));
        if (count > 0) {
          m_nms_out_file.write(reinterpret_cast<const char*>(m_nms_results[i].data()),
                               static_cast<std::streamsize>(m_nms_results[i].size() * sizeof(int32_t)));
        }
        if (!m_nms_out_file) {
          APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Unable to write NMS results for frame %zu to %s",
                  frame_count + i, m_nms_out_path.c_str());
          return vart_app_status::FAILURE;
        }
        APP_LOG(AppLogLevel::DEBUG, m_app_opt.app_log, "Wrote %d NMS detection(s) for frame %zu", count,
                frame_count + i);
      }
      std::cout << "Wrote NMS results for frames " << frame_count << "-" << (frame_count + actual_batch_size - 1)
                << " to file: " << m_nms_out_path << std::endl;
    } catch (const std::exception& e) {
      APP_LOG(AppLogLevel::ERROR, m_app_opt.app_log, "Error writing NMS results: %s", e.what());
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

      /* Convert the model outputs to the NMS input layout and run the nms_onnx
       * PL kernel, capturing the selected detections per frame. In benchmark
       * mode this is still run so its per-stage timings (conversion + PL
       * kernel) are measured; only the file writes below are skipped. */
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

    /* Per-stage breakdown of the ml_vart_plus_pl NMS datapath, each averaged
     * per frame. The data-conversion stage is timed in this class
     * (m_conv_us); the NMS PL sub-stages come from the pl_nms wrapper's
     * internal timers (accumulated across all runs):
     *   ML inference         : NPU run_inference() (== the average above)
     *   data-conversion (host): model output -> NMS boxes/scores layout
     *   data-transfer-to-PL  : host->PL input sync (boxes + scores)
     *   NMS PL kernel        : nms_onnx kernel launch + wait
     *   data-transfer-from-PL: PL->host result sync (num_selected + selected) */
    if (m_pl && m_pl_forward_frames > 0) {
      const double pl_denom = static_cast<double>(m_pl_forward_frames);
      const double ml_ms = avg_time_ms;
      const double conv_ms = (m_conv_us / pl_denom) / 1000.0;
      const double to_pl_ms = (m_pl->to_pl_us() / pl_denom) / 1000.0;
      const double pl_exec_ms = (m_pl->pl_exec_us() / pl_denom) / 1000.0;
      const double from_pl_ms = (m_pl->from_pl_us() / pl_denom) / 1000.0;

      std::cout << std::fixed << std::setprecision(3);
      std::cout << "Per-stage average (ms/frame):" << std::endl;
      std::cout << "  ML inference             : " << ml_ms << std::endl;
      std::cout << "  data-conversion (host)   : " << conv_ms << std::endl;
      std::cout << "  data-transfer-to-PL      : " << to_pl_ms << std::endl;
      std::cout << "  NMS PL kernel            : " << pl_exec_ms << std::endl;
      std::cout << "  data-transfer-from-PL    : " << from_pl_ms << std::endl;

      /* End-to-end total = sum of the five pipeline stages above: the full
       * per-frame latency of the NMS datapath (NPU inference + host data
       * conversion + NMS PL forward), as opposed to the "ML only" headline. */
      const double total_ms = ml_ms + conv_ms + to_pl_ms + pl_exec_ms + from_pl_ms;
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

    /* nms_onnx PL kernel configuration: the .xclbin containing the NMS
     * kernel, (optionally) the kernel name, device index, and the NMS scalar
     * parameters (ONNX NonMaxSuppression semantics). */
    if (auto pl_config = config.get_child_optional("pl-config")) {
      const auto& pl = pl_config.get();
      app_info.pl_xclbin = pl.get<std::string>("xclbin-location", app_info.pl_xclbin);
      app_info.pl_kernel = pl.get<std::string>("kernel-name", app_info.pl_kernel);
      app_info.pl_device_index = pl.get<unsigned int>("device-index", app_info.pl_device_index);
      app_info.nms_num_classes = pl.get<int>("num-classes", app_info.nms_num_classes);
      app_info.nms_max_out_per_class =
          pl.get<int>("max-output-boxes-per-class", app_info.nms_max_out_per_class);
      app_info.nms_iou_threshold = pl.get<float>("iou-threshold", app_info.nms_iou_threshold);
      app_info.nms_score_threshold = pl.get<float>("score-threshold", app_info.nms_score_threshold);
      app_info.nms_center_point_box = pl.get<int>("center-point-box", app_info.nms_center_point_box);
    }

    if (!app_info.dry_run && app_info.pl_xclbin.empty()) {
      APP_LOG(AppLogLevel::ERROR, app_info.app_log,
              "Missing required JSON field 'pl-config.xclbin-location' (path to the .xclbin containing the "
              "nms_onnx PL kernel).");
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

    /* Initialize the nms_onnx PL kernel used to post-process the outputs */
    if (vart_app_status::FAILURE == runner->init_pl_kernel()) {
      APP_LOG(AppLogLevel::ERROR, options.app_log, "Failed to initialize the nms_onnx PL kernel.");
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
