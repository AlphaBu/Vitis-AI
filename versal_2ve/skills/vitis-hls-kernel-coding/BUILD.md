# Build a Vitis HLS Kernel — `make compile` → `.xo`

This page gives ready-to-use **`Makefile`** + **`hls_config.cfg`** templates so a user can run:

```bash
make compile          # -> build/<KERNEL>.xo  (default platform: xilinx_vck190_base_202520_1)
```

The flow used is the **Vitis kernel compile** (`v++ -c`), which turns one HLS C/C++ top
function into a Xilinx Object (`.xo`) for the target **platform**. The `.xo` is later
linked into an `.xclbin` (`v++ -l`) — see [§ Linking](#optional-linking-to-xclbin).

> Refs: **UG1399** (Vitis HLS User Guide), **UG1393** (Vitis Application Acceleration),
> **UG1701** (Embedded Design Using Vitis).

---

## Roles of the two files

| File | Owns | Passed to |
|---|---|---|
| `Makefile` | Kernel name, source(s), **platform (default)**, target, output/dirs | drives `v++` |
| `hls_config.cfg` | HLS synthesis options (clock, uncertainty, extra directives) | `v++ -c --config` |

Keeping platform/target/kernel in the **Makefile** (as overridable variables) and the
HLS knobs in **`hls_config.cfg`** avoids duplicate-option conflicts and matches the
standard Vitis project layout.

---

## `Makefile`

```makefile
# =====================================================================
# Vitis HLS kernel build.   `make compile`  ->  build/<KERNEL>.xo
# Override any variable on the command line, e.g.:
#   make compile KERNEL=vadd SRC=vadd.cpp TARGET=hw_emu
#   make compile PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
# =====================================================================

# ---- User-overridable variables --------------------------------------
# IMPORTANT: no trailing "# ..." comment on an assignment line — make folds
# the spaces before the comment into the value and breaks paths/kernel name.
KERNEL    ?= vadd
# ^ top function name; must match the extern "C" function (passed as v++ -k)
SRC       ?= $(KERNEL).cpp
# ^ kernel source file(s); space-separate multiple files
PLATFORM  ?= xilinx_vck190_base_202520_1
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
```

Notes:
- `?=` makes every setting overridable: `make compile PLATFORM=... TARGET=hw_emu`.
- `-k $(KERNEL)` tells `v++` which top function to package; it must match the
  `extern "C"` function name in `$(SRC)`.
- Multiple sources: set `SRC = kernel.cpp helpers.cpp` (space-separated).
- `--save-temps` keeps the generated RTL/HLS project for inspection under `$(BUILD_DIR)/tmp`.
- Synthesis/utilization reports land in `$(BUILD_DIR)/reports`.

---

## `hls_config.cfg`

```ini
# =====================================================================
# v++ compile config — consumed by:  v++ -c --config hls_config.cfg
# Platform / target / kernel name are supplied by the Makefile
# (--platform / -t / -k), so they are intentionally NOT set here.
# See UG1393 (Vitis Compiler config) and UG1399 (HLS options).
# =====================================================================

# Target kernel clock frequency in MHz (global option = --kernel_frequency).
kernel_frequency=300

[hls]
# Extra HLS directives via a Tcl script run before synthesis
# (create_clock, config_*, set_directive_*), e.g.:
# pre_tcl=hls_directives.tcl
```

> **Important (verified on Vitis 2025.2):** in the `v++ -c` **kernel** flow the
> `[hls]` section does **not** accept `clock`/`clock_uncertainty` — those belong
> to the standalone `--mode hls` component flow. Set the kernel clock here with
> the global **`kernel_frequency=<MHz>`** (or the `[clock]` section / the
> `--kernel_frequency` flag). The only widely-supported `[hls]` key for the
> kernel flow is `pre_tcl` (point it at a Tcl file for `create_clock`,
> `config_*`, `set_directive_*`). Check your version with
> `v++ --help 2>&1 | grep -- '--hls\.'`.

Keep genuine **interface** and **performance** pragmas in the kernel `.cpp`
(portable, reviewable); use the cfg mainly for **project-level frequency** and
tool options. For fine HLS timing control, put `create_clock`/`config_*` in a
`pre_tcl` script.

---

## Quick start

```bash
# 1. Put kernel + these two files together:
#    vadd.cpp   Makefile   hls_config.cfg
# 2. Source the tool settings (adjust to your install):
source /opt/xilinx/Vitis/2025.2/settings64.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms   # where the .xpfm lives
# 3. Build:
make compile                     # -> build/vadd.xo
# 4. Emulation build instead of hardware:
make compile TARGET=hw_emu
```

If the platform is installed under a non-standard path, either export
`PLATFORM_REPO_PATHS` or pass a full path:
`make compile PLATFORM=/abs/path/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm`.

---

## (Optional) Linking to `.xclbin`

`make compile` stops at the `.xo`. To produce a loadable binary, link with a
connectivity config (`link.cfg`) — memory/stream mapping is a **link-time** choice:

```makefile
# Append to the Makefile if you also want to link:
LINK_CFG ?= link.cfg
XCLBIN   := $(BUILD_DIR)/$(KERNEL).xclbin

.PHONY: link
link: $(XCLBIN)                    ## Link <KERNEL>.xo into an .xclbin
$(XCLBIN): $(XO) $(LINK_CFG)
	$(VPP) -l $(VPP_FLAGS) --config $(LINK_CFG) -o $@ $(XO)
```

```ini
# link.cfg
[connectivity]
# Map kernel ports to memory banks / SLR (platform-dependent):
# sp=vadd_1.in1:DDR[0]
# sp=vadd_1.in2:DDR[0]
# sp=vadd_1.out:DDR[1]
# Free-running / kernel-to-kernel streams:
# stream_connect=producer_1.out:consumer_1.in
# nk=vadd:1                 # number of kernel instances
```

---

## Alternative: standalone HLS component (`v++ -c --mode hls`)

If you want the **HLS-component** flow (C-sim / synth / co-sim without a platform,
producing an `.xo` or IP), put everything in the config and use `--mode hls`:

```ini
# hls_component.cfg
part=xcvc1902-vsva2197-2MP-e-S      # a part, not a platform, for --mode hls
[hls]
flow_target=vitis                    # 'vitis' -> .xo ; 'vivado' -> IP
package.output.format=xo
syn.file=vadd.cpp
syn.top=vadd
tb.file=vadd_tb.cpp                  # optional C testbench
clock=3.33ns
```

```bash
v++ -c --mode hls --config hls_component.cfg --work_dir build_hls
```

Use the **Makefile + platform** flow above for kernels headed into an `.xclbin`;
use `--mode hls` when you just want to synthesize/verify a component against a raw part.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ERROR: platform not found` | `source settings64.sh`; set `PLATFORM_REPO_PATHS` or pass full `.xpfm` path |
| `Cannot find kernel <name>` | `-k` name ≠ `extern "C"` function name |
| Duplicate/conflicting option | Don't set `platform`/`target` in **both** cfg and Makefile |
| Very long build | `TARGET=hw` runs full synth; use `TARGET=sw_emu`/`hw_emu` while iterating |
| Bad timing/II in report | Inspect `build/reports`; apply pragmas (see OPTIMIZATION.md) |
