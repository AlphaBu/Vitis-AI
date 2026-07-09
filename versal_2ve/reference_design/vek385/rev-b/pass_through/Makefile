# =====================================================================
# Vitis HLS kernel build.   `make compile`  ->  build/<KERNEL>.xo
# Override any variable on the command line, e.g.:
#   make compile TARGET=hw_emu
#   make compile PLATFORM=/abs/path/to/vek385_base.xpfm
# =====================================================================

# ---- User-overridable variables --------------------------------------
# IMPORTANT: no trailing "# ..." comment on an assignment line — make folds
# the spaces before the comment into the value and breaks paths/kernel name.
KERNEL    ?= pass_through
# ^ top function name; must match the extern "C" function (passed as v++ -k)
SRC       ?= $(KERNEL).cpp
# ^ kernel source file(s); space-separate multiple files
PLATFORM  ?= ../hw/example_design_pfm_extensible.xsa
# ^ default platform (name found via PLATFORM_REPO_PATHS, or a full .xpfm path)
TARGET    ?= hw
# ^ hw | hw_emu | sw_emu
CFG       ?= hls_config.cfg
BUILD_DIR ?= build

# ---- Tools -----------------------------------------------------------
VPP       ?= v++

# ---- Derived ---------------------------------------------------------
XO        := $(BUILD_DIR)/$(KERNEL).xo

VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) \
             --save-temps \
             --temp_dir  $(BUILD_DIR)/tmp \
             --report_dir $(BUILD_DIR)/reports \
             --log_dir    $(BUILD_DIR)/logs

# ---- Targets ---------------------------------------------------------
.PHONY: compile clean cleanall help

compile: $(XO)                     ## Compile kernel source to <KERNEL>.xo

$(XO): $(SRC) $(CFG) | $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) --config $(CFG) -k $(KERNEL) -o $@ $(SRC)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:                             ## Remove build artifacts (keep .xo)
	rm -rf $(BUILD_DIR)/tmp $(BUILD_DIR)/logs $(BUILD_DIR)/reports \
	       *.log *.jou v++_* .Xil .run

cleanall: clean                    ## Remove everything including .xo
	rm -rf $(BUILD_DIR)

help:                              ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n",$$1,$$2}'
