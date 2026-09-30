#Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

import onnxruntime

provider_options_dict = {
    "config_file": 'vitisai_config.json',
    "cache_dir":   './',
    "cache_key":   'yolox_nano_onnx_pt_regular_conv_all',
    "ai_analyzer_visualization": True,
    "ai_analyzer_profiling": True,
    "log_level": 'info',
	"target": 'VAIML'
}
   
print(f"Creating ORT inference session ")
session = onnxruntime.InferenceSession(
    'onnx_model/yolox_nano_onnx_pt_regular_conv_all.onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict]
)   
 
