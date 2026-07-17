// ONNX NonMaxSuppression — Vitis HLS kernel
// Spec: https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html
//
// Compile-time capacities:
//   MAX_BOXES   : max spatial_dimension (boxes per batch)        default 4000
//   NUM_CLASSES : max number of score classes                    default 80
//   NUM_BATCHES : max batch size                                 default 1
//
// Inputs  (AXI master, DDR):
//   boxes[]            BF16 [NUM_BATCHES * MAX_BOXES * 4]
//                       center_point_box=0 : [y1,x1,y2,x2] per box
//                       center_point_box=1 : [cx,cy,w,h]   per box
//   scores[]           BF16 [NUM_BATCHES * NUM_CLASSES * MAX_BOXES]
//   Inputs are expanded to FP32 during load; internal storage and arithmetic
//   remain float.
//
// Scalar inputs (AXI-Lite):
//   num_batches, num_classes, num_boxes
//   max_out_per_class  (0 = unlimited → clamped to num_boxes)
//   iou_threshold, score_threshold (BF16, expanded to FP32 before compute)
//   center_point_box   (0 = corner, 1 = center)
//
// Outputs (AXI master, DDR):
//   selected_out[]     int32 [NUM_BATCHES * NUM_CLASSES * MAX_BOXES * 3]
//                       each triplet: [batch_idx, class_idx, box_idx]
//   num_selected[0]    int32 number of valid selected triplets

#include <ap_int.h>
#include <ap_float.h>
#include <hls_math.h>
#include <stdint.h>

// ── compile-time capacities ──────────────────────────────────────────────────
#ifndef MAX_BOXES
#define MAX_BOXES   4000
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 80
#endif
#ifndef NUM_BATCHES
#define NUM_BATCHES 1
#endif

static const int MAX_SELECTED = NUM_BATCHES * NUM_CLASSES * MAX_BOXES;
using bf16_t = ap_float<16, 8>;

// ── helpers ──────────────────────────────────────────────────────────────────

// Expand a BF16 top-level value to FP32 at the kernel boundary. All on-chip
// storage and arithmetic below remain float.
static float bf16_to_float(bf16_t value)
{
#pragma HLS INLINE
    return static_cast<float>(value);
}

// Convert center [cx,cy,w,h] → corner [y1,x1,y2,x2]
static void center_to_corner(float cx, float cy, float w, float h,
                              float &y1, float &x1, float &y2, float &x2)
{
    float hw = w * 0.5f;
    float hh = h * 0.5f;
    x1 = cx - hw;  x2 = cx + hw;
    y1 = cy - hh;  y2 = cy + hh;
}

// True iff IoU(box_i, box_j) > thr, both in corner [y1,x1,y2,x2] format.
// Divide-free: since union_area > 0 in the relevant branch, the test
// inter_area/union_area > thr is equivalent to inter_area > thr*union_area.
static bool iou_exceeds(const float box_i[4], const float box_j[4], float thr)
{
    float iy1 = box_i[0], ix1 = box_i[1], iy2 = box_i[2], ix2 = box_i[3];
    float jy1 = box_j[0], jx1 = box_j[1], jy2 = box_j[2], jx2 = box_j[3];

    float inter_y1 = (iy1 > jy1) ? iy1 : jy1;
    float inter_x1 = (ix1 > jx1) ? ix1 : jx1;
    float inter_y2 = (iy2 < jy2) ? iy2 : jy2;
    float inter_x2 = (ix2 < jx2) ? ix2 : jx2;

    float inter_h = inter_y2 - inter_y1;
    float inter_w = inter_x2 - inter_x1;
    if (inter_h <= 0.0f || inter_w <= 0.0f) return false;

    float inter_area = inter_h * inter_w;
    float area_i = (iy2 - iy1) * (ix2 - ix1);
    float area_j = (jy2 - jy1) * (jx2 - jx1);
    float union_area = area_i + area_j - inter_area;
    if (union_area <= 0.0f) return false;
    return inter_area > thr * union_area;
}

// ── on-chip scratch (PIPO between load/compute/store) ────────────────────────
// Declared static so HLS allocates BRAM/URAM; sizes are worst-case maximums.
static float  pipo_boxes  [NUM_BATCHES][MAX_BOXES][4];
static float  pipo_scores [NUM_BATCHES][NUM_CLASSES][MAX_BOXES];
static int32_t pipo_selected[MAX_SELECTED][3];   // [batch, class, box]

// ── load_input ───────────────────────────────────────────────────────────────
static void load_input(
    const bf16_t *boxes_in,
    const bf16_t *scores_in,
    int nb, int nc, int nx,
    int center_point_box)
{
    // Burst-read boxes: layout [batch][box][4]
    load_boxes:
    for (int b = 0; b < nb; b++) {
        for (int n = 0; n < nx; n++) {
            int base = (b * MAX_BOXES + n) * 4;
            float raw[4];
            raw[0] = bf16_to_float(boxes_in[base+0]);
            raw[1] = bf16_to_float(boxes_in[base+1]);
            raw[2] = bf16_to_float(boxes_in[base+2]);
            raw[3] = bf16_to_float(boxes_in[base+3]);
            if (center_point_box) {
                float y1, x1, y2, x2;
                center_to_corner(raw[0], raw[1], raw[2], raw[3], y1, x1, y2, x2);
                pipo_boxes[b][n][0] = y1;
                pipo_boxes[b][n][1] = x1;
                pipo_boxes[b][n][2] = y2;
                pipo_boxes[b][n][3] = x2;
            } else {
                pipo_boxes[b][n][0] = raw[0];
                pipo_boxes[b][n][1] = raw[1];
                pipo_boxes[b][n][2] = raw[2];
                pipo_boxes[b][n][3] = raw[3];
            }
        }
    }

    // Burst-read scores: layout [batch][class][box]
    load_scores:
    for (int b = 0; b < nb; b++) {
        for (int c = 0; c < nc; c++) {
            for (int n = 0; n < nx; n++) {
                pipo_scores[b][c][n] = bf16_to_float(
                    scores_in[(b * NUM_CLASSES + c) * MAX_BOXES + n]);
            }
        }
    }
}

// ── score → sortable key ──────────────────────────────────────────────────────
// Map an IEEE-754 float to a uint32 whose *ascending* unsigned order equals the
// *descending* float order. Radix-sorting these keys ascending therefore yields
// boxes highest-score-first, as ONNX NMS requires. Handles negatives correctly.
static uint32_t score_key_desc(float s)
{
    union { float f; uint32_t u; } c;
    c.f = s;
    uint32_t u = c.u;
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // float order == unsigned order
    return ~u;                                         // invert: asc key == desc score
}

// ── one stable LSD radix pass over an 8-bit digit ─────────────────────────────
static void radix_pass(const uint32_t *src_key, const int *src_idx,
                       uint32_t *dst_key, int *dst_idx,
                       int n, int shift)
{
    int count[256];
    radix_init:
    for (int d = 0; d < 256; d++) {
#pragma HLS PIPELINE II=1
        count[d] = 0;
    }

    radix_count:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        count[(src_key[i] >> shift) & 0xFF]++;
    }

    int sum = 0;
    radix_prefix:
    for (int d = 0; d < 256; d++) {
#pragma HLS PIPELINE II=1
        int c = count[d];
        count[d] = sum;
        sum += c;
    }

    radix_scatter:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        int d   = (src_key[i] >> shift) & 0xFF;
        int pos = count[d]++;
        dst_key[pos] = src_key[i];
        dst_idx[pos] = src_idx[i];
    }
}

// ── sort candidate indices by descending score — O(n) 4-pass LSD radix ────────
// Ping-pongs between cand_idx and a scratch buffer; the even pass count leaves
// the sorted result back in cand_idx.
static void sort_candidates(int *cand_idx, const float *cand_score, int num_cand)
{
    uint32_t keyA[MAX_BOXES], keyB[MAX_BOXES];
    int      idxB[MAX_BOXES];

    radix_build:
    for (int i = 0; i < num_cand; i++) {
#pragma HLS PIPELINE II=1
        keyA[i] = score_key_desc(cand_score[i]);
    }

    radix_pass(keyA, cand_idx, keyB, idxB,     num_cand, 0);
    radix_pass(keyB, idxB,     keyA, cand_idx, num_cand, 8);
    radix_pass(keyA, cand_idx, keyB, idxB,     num_cand, 16);
    radix_pass(keyB, idxB,     keyA, cand_idx, num_cand, 24);
}

// ── compute ──────────────────────────────────────────────────────────────────
// Per class, filter and sort must complete before greedy suppression. The outer
// greedy loop also has a true dependency: each accepted box updates suppression
// state before the next accepted box can be determined.
static void compute(
    int nb, int nc, int nx,
    int max_out,
    float iou_thr, float score_thr,
    int32_t *num_selected_out)
{
    int32_t total = 0;

    for (int b = 0; b < nb; b++) {
        for (int c = 0; c < nc; c++) {

            // ── 1. filter by score_threshold ─────────────────────────────
            // Collect candidate indices above threshold
            int  cand_idx[MAX_BOXES];
            float cand_score[MAX_BOXES];
            int  num_cand = 0;
#pragma HLS ARRAY_PARTITION variable=cand_idx cyclic factor=8

            filter_loop:
            for (int n = 0; n < nx; n++) {
                float s = pipo_scores[b][c][n];
                if (s > score_thr) {
                    cand_idx[num_cand]   = n;
                    cand_score[num_cand] = s;
                    num_cand++;
                }
            }

            // ── 2. sort candidates by score descending (O(n) LSD radix) ──
            // Linear in num_cand (4 passes over 8-bit digits); the greedy
            // stage below reads only cand_idx, so we sort indices by key.
            sort_candidates(cand_idx, cand_score, num_cand);

            // Gather box coordinates into sorted-candidate order.  The greedy
            // scan visits consecutive ranks, so cyclic partitioning gives the
            // eight unrolled IoU lanes independent coordinate banks instead of
            // contending for the randomly indexed pipo_boxes memories.
            float sorted_y1[MAX_BOXES];
            float sorted_x1[MAX_BOXES];
            float sorted_y2[MAX_BOXES];
            float sorted_x2[MAX_BOXES];
            // Suppression state is indexed by position in the sorted candidate
            // list.  Initialize only valid ranks while gathering coordinates,
            // avoiding a separate MAX_BOXES-cycle clear before greedy NMS.
            bool suppressed_rank[MAX_BOXES];
#pragma HLS ARRAY_PARTITION variable=sorted_y1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=sorted_x1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=sorted_y2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=sorted_x2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=suppressed_rank cyclic factor=8

            gather_boxes:
            for (int k = 0; k < num_cand; k++) {
#pragma HLS PIPELINE II=1
                int box = cand_idx[k];
                sorted_y1[k] = pipo_boxes[b][box][0];
                sorted_x1[k] = pipo_boxes[b][box][1];
                sorted_y2[k] = pipo_boxes[b][box][2];
                sorted_x2[k] = pipo_boxes[b][box][3];
                suppressed_rank[k] = false;
            }

            // ── 3. greedy NMS ─────────────────────────────────────────────
            int  class_selected = 0;
            int  effective_max  = (max_out == 0 || max_out > nx) ? nx : max_out;

            greedy_loop:
            for (int i = 0; i < num_cand; i++) {
                if (class_selected >= effective_max) break;

                int bi = cand_idx[i];
                if (suppressed_rank[i]) continue;

                // Accept this box
                if (total < MAX_SELECTED) {
                    pipo_selected[total][0] = (int32_t)b;
                    pipo_selected[total][1] = (int32_t)c;
                    pipo_selected[total][2] = (int32_t)bi;
                    total++;
                    class_selected++;
                }

                float selected_box[4];
#pragma HLS ARRAY_PARTITION variable=selected_box complete
                selected_box[0] = sorted_y1[i];
                selected_box[1] = sorted_x1[i];
                selected_box[2] = sorted_y2[i];
                selected_box[3] = sorted_x2[i];

                // Suppress overlapping boxes
                iou_loop:
                for (int j = i + 1; j < num_cand; j++) {
#pragma HLS UNROLL factor=8
                    if (suppressed_rank[j]) continue;

                    float candidate_box[4];
#pragma HLS ARRAY_PARTITION variable=candidate_box complete
                    candidate_box[0] = sorted_y1[j];
                    candidate_box[1] = sorted_x1[j];
                    candidate_box[2] = sorted_y2[j];
                    candidate_box[3] = sorted_x2[j];

                    if (iou_exceeds(selected_box, candidate_box, iou_thr))
                        suppressed_rank[j] = true;
                }
            }
        }
    }

    *num_selected_out = total;
}

// ── store_output ─────────────────────────────────────────────────────────────
static void store_output(int32_t *out, int32_t num_sel)
{
    store_loop:
    for (int i = 0; i < num_sel; i++) {
        out[i*3+0] = pipo_selected[i][0];
        out[i*3+1] = pipo_selected[i][1];
        out[i*3+2] = pipo_selected[i][2];
    }
}

// ── top-level ─────────────────────────────────────────────────────────────────
void nms_onnx(
    const bf16_t *boxes_in,        // m_axi: [nb * MAX_BOXES * 4]
    const bf16_t *scores_in,       // m_axi: [nb * NUM_CLASSES * MAX_BOXES]
    int32_t      *selected_out,    // m_axi: [MAX_SELECTED * 3]
    int           num_batches,
    int           num_classes,
    int           num_boxes,
    int           max_out_per_class,
    bf16_t        iou_threshold,
    bf16_t        score_threshold,
    int           center_point_box,
    int32_t      *num_selected)    // AXI-Lite scalar output
{
#pragma HLS INTERFACE m_axi port=boxes_in      offset=slave bundle=gmem0 depth=(NUM_BATCHES*MAX_BOXES*4)
#pragma HLS INTERFACE m_axi port=scores_in     offset=slave bundle=gmem1 depth=(NUM_BATCHES*NUM_CLASSES*MAX_BOXES)
#pragma HLS INTERFACE m_axi port=selected_out  offset=slave bundle=gmem2 depth=(MAX_SELECTED*3)
#pragma HLS INTERFACE m_axi port=num_selected  offset=slave bundle=gmem3 depth=1
#pragma HLS INTERFACE s_axilite port=num_batches
#pragma HLS INTERFACE s_axilite port=num_classes
#pragma HLS INTERFACE s_axilite port=num_boxes
#pragma HLS INTERFACE s_axilite port=max_out_per_class
#pragma HLS INTERFACE s_axilite port=iou_threshold
#pragma HLS INTERFACE s_axilite port=score_threshold
#pragma HLS INTERFACE s_axilite port=center_point_box
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    int32_t ns = 0;

    load_input(boxes_in, scores_in,
               num_batches, num_classes, num_boxes,
               center_point_box);

    compute(num_batches, num_classes, num_boxes,
            max_out_per_class,
            bf16_to_float(iou_threshold),
            bf16_to_float(score_threshold),
            &ns);

    store_output(selected_out, ns);

    *num_selected = ns;
}
