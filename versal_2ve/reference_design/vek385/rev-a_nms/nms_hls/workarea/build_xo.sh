#!/bin/bash
# Build nms_onnx.xo with the full Vitis 2025.2 v++ (HLS kernel flow).
set -e
cd /proj/xcohdstaff3/brucey/nobkup/VAIML/Vitis-AI/versal_2ve/reference_design/vek385/rev-a/nms_hls/workarea

ROOT=/proj/xbuilds/SWIP/2025.2_1114_2157/installs/lin64/2025.2
VITIS=$ROOT/Vitis
# Source full settings so XILINX_VIVADO / XILINX_VITIS etc. are set (v++ requires them).
source "$ROOT/Vivado/settings64.sh"
source "$VITIS/settings64.sh"

echo "== v++ =="; "$VITIS/bin/v++" --version 2>&1 | head -2
make compile VPP="$VITIS/bin/v++"
echo "== .xo =="; ls -la build/nms_onnx.xo
