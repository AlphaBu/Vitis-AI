This folder holds shared references used across `examples/cpp_examples` apps (JSON schemas, placement policy, and compilation guides).

- **[glossary.md](glossary.md)** — acronyms (IFM, OFM, NPU, EP, ORT, VART, etc.) used in example READMEs
- **[mixed_precision.md](mixed_precision.md)** — quantize, compile, and run INT8-head + BF16/FP16-tail models
- **[multi_tenancy.md](multi_tenancy.md)** — data vs. tensor parallelism, spatial vs. temporal multi-tenancy, and NPU column layouts
- **[auto_placement_policy.md](auto_placement_policy.md)** — how the runtime places models on the NPU when `start-column` is omitted
- **[runner_options.md](runner_options.md)** — `runner-options` schema used by VART-ML sample JSON
- **[preprocessing_config.md](preprocessing_config.md)** — `preprocess-config` schema (colour format, resize, HLS kernel)
- **[postprocessing_config.md](postprocessing_config.md)** — `postprocess-config` types and fields
- **[metaconvert_config.md](metaconvert_config.md)** — overlay / metaconvert JSON fields
- **[npu_format_selection_guide.md](npu_format_selection_guide.md)** — NPU tensor formats and layout selection
