# YOLOX-Nano INT8 + `pass_through` PL — full COCO val2017 evaluation pipeline

End-to-end recipe for measuring COCO mAP of the YOLOX-Nano INT8 model run through
`ml_vart_plus_pl` on the VEK385: the NPU runs the `.rai` model, every output tensor
is forwarded byte-for-byte through the `pass_through` PL kernel, and the PL outputs
(OFM `.bin` files) are de-tiled, dequantized, post-processed and scored with
pycocotools.

Because `pass_through` is an identity copy, the PL stage is numerically a no-op; the
mAP measured is the NPU model's accuracy exercised over the full `ml_vart_plus_pl` +
PL datapath.

**Last full-run result (5000 images, 330418 detections):**

| Metric            | Value  |
|-------------------|--------|
| mAP@[.5:.95]      | 0.209  |
| mAP@.5            | 0.367  |
| mAP@.75           | 0.216  |
| AR@100            | 0.366  |

Reference NPU YOLOX-Nano ≈ 0.190 / 0.343 — the datapath is confirmed correct.

---

## 0. Components & fixed paths

| Item              | Path                                                                                             |
|-------------------|--------------------------------------------------------------------------------------------------|
| Runner binary     | `/wrk/xcohdnobkup3/brucey/VAIML/Vitis-AI/versal_2ve/examples/cpp_examples/ml_vart_plus_pl/ml_vart_plus_pl` (aarch64, 4.1M) |
| Eval harness      | `/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_int8_testPL/coco_pl_eval/`                      |
| Model `.rai`      | `/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_int8_testPL/yolox_nano_onnx_pt_regular_conv_all/yolox_nano_onnx_pt_regular_conv_all.rai` (20M, self-contained) |
| Packed work dir   | `/wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/full/` (10 chunks × 500 = 5000 images)                |
| COCO dataset      | `/proj/xcohdstaff3/brucey/nobkup/VAIML/datasets/coco/`                                            |
| xclbin (on board) | `/run/media/mmcblk0p1/x_plus_ml.xclbin` (`pass_through_1` + `image_processing_1` + AIE)           |

Board: **VEK385 @ 10.25.38.208**, root login has an EMPTY password.

Host Python env:

```bash
VENV=/proj/xcohdstaff3/brucey/nobkup/VAIML/ryzen_ai-1.7.1/venv/bin/python3   # cv2 / numpy / onnxruntime
PYLIBS=/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_1.5ms/pylibs         # pycocotools (NOT in venv)
EVAL=/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_int8_testPL/coco_pl_eval
```

---

## Key insight: run over NFS, not scp

The board mounts the NFS export `10.25.34.124:/wrk/xcohdnobkup3/brucey/VAIML` at
`/mnt`, so:

```
host  /wrk/xcohdnobkup3/brucey/VAIML/...   ==   board  /mnt/...
```

That means the packed inputs/configs, the runner binary and the model `.rai` are all
directly visible on the board with **no copying**, and OFMs written by the board land
straight back on the host filesystem. **Do NOT scp the 3.46 GB of inputs** — board-side
ssh crypto sustains only ~0.15 MB/s and a large sustained scp has crashed/rebooted the
board. Over NFS the whole run is ~54 s/chunk, ~9 min total.

Mapping used here: host `/wrk/.../coco_pl_run/full` == board `/mnt/coco_pl_run/full`.

---

## Model quant constants (baked into `yolox_pl_common.py`)

* input  `DequantizeLinear` scale **4.0**, zp 0  → `q = clip(round(px/4.0), -128, 127)` int8
* output `DequantizeLinear` scale **0.125**, zp 0 → `float = q * 0.125`
* input BO layout: NHWC, `H×W×4` int8; ch0-2 = R/G/B-order preproc, ch3 = 0; per-frame `416*416*4 = 692224 B`
* runner input tensor name: `onnx::DequantizeLinear_229` (HW layout HCWNC4 `[416,1,416,1,4]`)
* OFM tile per head: `(H, 11, W, 1, 8)` int8 (HCWNC8); channel = `co*8 + ci`, keep `[:85]`; heads 52/26/13
* postprocess: sort heads by spatial size (stride 8/16/32), sigmoid `[4:]`, conf 0.01, nms 0.65

---

## Stage 1 — pack COCO → input BOs + configs (host, one-time)

Already done for the full set (present under `coco_pl_run/full/`). To regenerate:

```bash
PYTHONPATH=$PYLIBS $VENV $EVAL/pack_coco.py --out-dir /wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/full
# subset:  --limit 500 ;  chunk size:  --chunk-size 500 (default)
```

Produces per chunk `input_chunk{c}.bin` (346112000 B = 500 × 692224), `config_chunk{c}.json`,
plus a single `meta.json` (per-frame `img_id`/`ratio`, conf/nms, ann-file path). Full set ≈ 3.46 GB.

Each `config_chunk{c}.json` looks like:

```json
{
    "inference-config": {
        "model-file": "yolox_nano_onnx_pt_regular_conv_all/yolox_nano_onnx_pt_regular_conv_all.rai",
        "runner-options": { "log-level": "WARNING", "ai-analyzer-profiling": false }
    },
    "ifms-config": [
        { "name": "onnx::DequantizeLinear_229", "file": "input_chunk0.bin" }
    ],
    "ofms-dir": "ofm_chunk0",
    "pl-config": {
        "xclbin-location": "/run/media/mmcblk0p1/x_plus_ml.xclbin",
        "kernel-name": "pass_through",
        "device-index": 1
    }
}
```

`model-file`, `ifms-config.file` and `ofms-dir` are **relative** — resolved from the
runner's CWD, which must be the work dir. `device-index: 1` is required (NPU/AIE = XRT
device 0, PL region = device 1).

## Stage 1b — stage the `.rai` into the work dir (host)

The `model-file` path is relative, so the self-contained `.rai` must sit under the work
dir at that relative path. `/proj` is not NFS-mounted on the board, so copy it in:

```bash
SRC=/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_int8_testPL/yolox_nano_onnx_pt_regular_conv_all/yolox_nano_onnx_pt_regular_conv_all.rai
DST=/wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/full/yolox_nano_onnx_pt_regular_conv_all
mkdir -p "$DST" && cp "$SRC" "$DST/"
```

---

## Stage 2 — run on the board over NFS

Prereqs to verify first (fast checks; do NOT `find` over `/mnt`, it recurses the whole export):

```bash
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o ConnectTimeout=40"
ssh $SSHO -l root 10.25.38.208 '
  ls /mnt/coco_pl_run/full/ | head
  ls -la /mnt/Vitis-AI/versal_2ve/examples/cpp_examples/ml_vart_plus_pl/ml_vart_plus_pl
  ls -la /run/media/mmcblk0p1/x_plus_ml.xclbin
' </dev/null 2>&1 | grep -v -Ei "kwallet|qdbus|kf.coreaddons|Warning: Permanently"
```

`-o UserKnownHostsFile=/dev/null` is required because the board regenerates its host key on
every reboot. Filter the `grep -v` noise (kwallet/QDBus) from non-interactive ssh.

Run all 10 chunks in a single board-side loop. The binary is **not** in the work dir, so
call it by absolute path while keeping CWD = the work dir (for the relative config paths):

```bash
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o ConnectTimeout=40"
BIN=/mnt/Vitis-AI/versal_2ve/examples/cpp_examples/ml_vart_plus_pl/ml_vart_plus_pl
ssh $SSHO -l root 10.25.38.208 "
  export LD_LIBRARY_PATH=/usr/lib/python3.12/site-packages/flexmlrt/lib:\$LD_LIBRARY_PATH
  cd /mnt/coco_pl_run/full || exit 9
  for c in 0 1 2 3 4 5 6 7 8 9; do
    echo \"=== chunk \$c \$(date +%H:%M:%S) ===\"
    rm -rf ofm_chunk\$c; mkdir -p ofm_chunk\$c
    $BIN --app-config config_chunk\$c.json > /tmp/chunk\$c.log 2>&1
    echo \"  rc=\$? files=\$(ls ofm_chunk\$c/ | wc -l) tail=\$(tail -1 /tmp/chunk\$c.log)\"
  done
  echo \"ALL DONE \$(date +%H:%M:%S)\"
" </dev/null 2>&1 | grep -v -Ei "kwallet|qdbus|kf.coreaddons|Warning: Permanently"
```

`LD_LIBRARY_PATH` must include the flexmlrt lib or `vart::Runner` fails with "Cannot open
library libflexmlrt.so". Each chunk prints `Run completed successfully.` and writes 3 OFM
head files into `ofm_chunk{c}/`:

```
ofm_chunk{c}/infer_out0-int8_52x11x52x1x8_onnx::DequantizeLinear_2040.bin
ofm_chunk{c}/infer_out1-int8_26x11x26x1x8_onnx::DequantizeLinear_2266.bin
ofm_chunk{c}/infer_out2-int8_13x11x13x1x8_onnx::DequantizeLinear_2492.bin
```

All frames of a chunk are concatenated per head (~150M/chunk). Because of NFS, they are
immediately readable on the host at `/wrk/.../coco_pl_run/full/ofm_chunk{c}/`.

Verify on the host:

```bash
for c in 0 1 2 3 4 5 6 7 8 9; do echo "chunk$c: $(ls /wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/full/ofm_chunk$c/ | wc -l)"; done
```

---

## Stage 3 — de-tile + postprocess + COCO mAP (host)

Use a wrapper script (`/wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/run_eval.sh`) because
this shell env mangles inline multi-line `VAR=...; cmd` chains and bare `cd`:

```bash
#!/usr/bin/env bash
set -euo pipefail
EVAL=/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_int8_testPL/coco_pl_eval
PYLIBS=/proj/xcohdstaff3/brucey/nobkup/VAIML/yolox_nano_1.5ms/pylibs
VENV=/proj/xcohdstaff3/brucey/nobkup/VAIML/ryzen_ai-1.7.1/venv/bin/python3
WORK=/wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/full
export PYTHONPATH="$PYLIBS:$EVAL"
cd "$EVAL"
exec "$VENV" unpack_eval.py --work-dir "$WORK" --compute-map
```

```bash
bash /wrk/xcohdnobkup3/brucey/VAIML/coco_pl_run/run_eval.sh
```

`unpack_eval.py` reads `meta.json`, de-tiles+dequantizes each frame's 3 heads from the OFM
`.bin`s, runs the YOLOX postprocess with each frame's saved `ratio`, assembles COCO
detections and prints mAP@[.5:.95] / mAP@.5. Takes ~1 min for 5000 images.

Save detections without scoring: `unpack_eval.py --work-dir <work> --dump-results dets.json`.

---

## Stage 2b — per-stage timing benchmark (board, optional)

`ml_vart_plus_pl --benchmark` runs the full ML→PL datapath but writes no OFM files, and
reports the average per-frame latency of each stage separately:

* **ML inference** — NPU `run_inference()`
* **data-transfer-to-PL** — host→PL input staging (memcpy-in + `sync TO_DEVICE`); **0 for
  the zero-copy path**, which is the stage the dma-buf conversion eliminates
* **PL inference** — `pass_through` kernel launch + wait
* **data-transfer-from-PL** — PL→host result copy-back (`sync FROM_DEVICE` + memcpy-out)

The ML→PL transfer defaults to **zero-copy** (NPU output tensors exported as dma-buf and
imported into the PL device, so `pass_through` reads them in place). Set env
**`PL_ZEROCOPY=0`** to select the original host-copy path for an A/B comparison.

Benchmark all 10 chunks in both modes (5000 frames per mode) over NFS:

```bash
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o ConnectTimeout=40"
BIN=/mnt/Vitis-AI/versal_2ve/examples/cpp_examples/ml_vart_plus_pl/ml_vart_plus_pl
ssh $SSHO -l root 10.25.38.208 "
  export LD_LIBRARY_PATH=/usr/lib/python3.12/site-packages/flexmlrt/lib:\$LD_LIBRARY_PATH
  cd /mnt/coco_pl_run/full || exit 9
  for mode in ZC HC; do
    if [ \$mode = HC ]; then PFX='PL_ZEROCOPY=0'; else PFX=''; fi
    for c in 0 1 2 3 4 5 6 7 8 9; do
      echo \"--- \$mode chunk\$c ---\"
      env \$PFX $BIN --app-config config_chunk\$c.json --benchmark --runs 1 2>&1 \
        | grep -Ei 'ML inference|data-transfer|PL inference'
    done
  done
" </dev/null 2>&1 | grep -v -Ei "kwallet|qdbus|kf.coreaddons|Warning: Permanently"
```

**Last full-run result (5000 images = 10 chunks × 500, mean over chunks):**

| Stage                 | zero-copy | host-copy | saved  |
|-----------------------|-----------|-----------|--------|
| ML inference          | 1.381     | 1.381     | —      |
| data-transfer-to-PL   | **0.000** | **0.309** | 0.309  |
| PL inference          | 0.325     | 0.327     | —      |
| data-transfer-from-PL | 0.314     | 0.309     | —      |

*(ms/frame)*. Zero-copy removes the entire ML→PL input transfer (~0.31 ms/frame → 0), with
ML/PL inference and the from-PL copy-back unchanged. mAP is unaffected (0.209/0.367),
confirming the datapath still produces byte-correct OFMs.

---

## Host-only validation (no board)

```bash
PYTHONPATH=$PYLIBS $VENV $EVAL/validate_host.py --detile-check          # input layout + OFM de-tile vs real HW dumps
PYTHONPATH=$PYLIBS $VENV $EVAL/validate_host.py --cpu-map --limit 100   # CPU end-to-end mAP on a subset
```

`--detile-check` head0 mean abs err ≈ 0.074 (genuine NPU-vs-CPU numeric diff, layout is
correct); `--cpu-map --limit 100` ≈ 0.30.

---

## Troubleshooting / gotchas

* **scp is unusably slow** (~0.15 MB/s, crypto-bound on board CPU) and a large sustained
  transfer has rebooted the board. Always prefer the NFS path when `/mnt` is live.
* **NFS can go stale** — symptom: `ls /mnt` hangs. Then either wait for it to recover or
  fall back to scp to board-local `/home/root` (~24 G free). The board also has a volatile
  ramdisk-root state after some reboots in which staging is wiped and the SD/xclbin are
  unmounted; wait for it to return to the ext4 `/dev/sdd3` root.
* **ssh host key changes on reboot** → use `-o UserKnownHostsFile=/dev/null` (and
  `ssh-keygen -R 10.25.38.208` if a stale key is cached).
* **`./ml_vart_plus_pl: No such file or directory`** — the binary is not in the work dir;
  invoke it by absolute path (`/mnt/Vitis-AI/.../ml_vart_plus_pl`).
* **`Cannot open library libflexmlrt.so`** — missing `LD_LIBRARY_PATH` flexmlrt entry.
* Confirm HW layouts any time with `ml_vart_plus_pl --get-model-info <rai>`.
