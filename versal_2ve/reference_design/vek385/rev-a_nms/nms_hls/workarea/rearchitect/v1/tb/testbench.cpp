// Testbench for the BF16-input, FP32-internal nms_onnx HLS kernel.
// Verifies corner/center formats and two 6300-box workload shapes against a
// software reference with the same BF16 input quantization.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdint.h>
// Kernel declaration — HLS links kernel_hls.cpp separately
#include "../src/kernel_hls.h"

typedef bf16_t bf16;
// The DUT expands BF16 inputs to float and performs all geometry in FP32.
typedef float coord_t;

// ── reference NMS (used as golden) ────────────────────────────────────────────
// Boxes and thresholds are first quantized through BF16, then expanded to FP32
// like the DUT. Scores stay BF16 in the reference so filtering and stable score
// ordering use the same quantized values as the kernel inputs.
static bool ref_iou_exceeds(const coord_t *bi, const coord_t *bj, coord_t thr)
{
    coord_t iy1=bi[0],ix1=bi[1],iy2=bi[2],ix2=bi[3];
    coord_t jy1=bj[0],jx1=bj[1],jy2=bj[2],jx2=bj[3];
    coord_t ay1=(iy1>jy1)?iy1:jy1, ax1=(ix1>jx1)?ix1:jx1;
    coord_t ay2=(iy2<jy2)?iy2:jy2, ax2=(ix2<jx2)?ix2:jx2;
    coord_t ih=ay2-ay1, iw=ax2-ax1;
    if(ih<=0||iw<=0) return false;
    typedef float area_t;
    area_t ia=(area_t)ih*(area_t)iw;
    area_t ua=(area_t)(iy2-iy1)*(area_t)(ix2-ix1)
             +(area_t)(jy2-jy1)*(area_t)(jx2-jx1)-ia;
    if(ua<=0) return false;
    return ia > (area_t)thr*ua;
}

static int ref_nms(
    const coord_t *boxes,  // [nb][nx][4] corner format
    const bf16    *scores, // [nb][nc][nx]
    int nb, int nc, int nx,
    int max_out, bf16 iou_thr, bf16 score_thr,
    int32_t *out)        // [result][3]
{
    int total=0;
    coord_t iou_thr_f=(coord_t)iou_thr;
    for(int b=0;b<nb;b++){
        for(int c=0;c<nc;c++){
            // collect candidates
            int idx[MAX_BOXES]; bf16 sc[MAX_BOXES]; int nk=0;
            for(int n=0;n<nx;n++){
                bf16 s=scores[(b*NUM_CLASSES+c)*MAX_BOXES+n];
                if(s>score_thr){ idx[nk]=n; sc[nk]=s; nk++; }
            }
            // insertion sort descending (stable, matches kernel radix on ties)
            for(int i=1;i<nk;i++){
                bf16 ks=sc[i]; int ki=idx[i]; int j=i-1;
                while(j>=0&&sc[j]<ks){sc[j+1]=sc[j];idx[j+1]=idx[j];j--;}
                sc[j+1]=ks; idx[j+1]=ki;
            }
            bool sup[MAX_BOXES]={};
            int got=0;
            int eff=(max_out==0||max_out>nx)?nx:max_out;
            for(int i=0;i<nk&&got<eff;i++){
                int bi=idx[i]; if(sup[bi]) continue;
                out[total*3+0]=(int32_t)b;
                out[total*3+1]=(int32_t)c;
                out[total*3+2]=(int32_t)bi;
                total++; got++;
                for(int j=i+1;j<nk;j++){
                    int bj=idx[j]; if(sup[bj]) continue;
                    if(ref_iou_exceeds(&boxes[(b*MAX_BOXES+bi)*4],
                                       &boxes[(b*MAX_BOXES+bj)*4], iou_thr_f))
                        sup[bj]=true;
                }
            }
        }
    }
    return total;
}

// ── test helpers ─────────────────────────────────────────────────────────────
static bool triplet_match(const int32_t *a, int na,
                           const int32_t *b, int nb)
{
    if(na!=nb) return false;
    // order must match (both are output in same greedy order)
    for(int i=0;i<na;i++)
        if(a[i*3]!=b[i*3]||a[i*3+1]!=b[i*3+1]||a[i*3+2]!=b[i*3+2])
            return false;
    return true;
}

// ── test case 1: corner format, 1 batch, 1 class, 6 boxes ───────────────────
static bool test_corner()
{
    static bf16 boxes[1*MAX_BOXES*4]={};
    static coord_t boxes_ref[1*MAX_BOXES*4]={};
    static bf16 scores[1*NUM_CLASSES*MAX_BOXES]={};
    static int32_t dut_out[MAX_SELECTED*3]={};
    static int32_t ref_out[MAX_SELECTED*3]={};

    // 6 boxes [y1,x1,y2,x2]
    float raw[6][4]={
        {0,0,10,10},{1,1,11,11},{0,0,10,10},
        {5,5,15,15},{6,6,16,16},{100,100,110,110}
    };
    float sc[6]={0.9f,0.75f,0.6f,0.95f,0.5f,0.3f};
    for(int i=0;i<6;i++){
        for(int k=0;k<4;k++){
            boxes[i*4+k]=bf16(raw[i][k]);
            // Reference mirrors the DUT path: float → BF16 port → FP32.
            boxes_ref[i*4+k]=coord_t((float)bf16(raw[i][k]));
        }
        scores[i]=bf16(sc[i]);
    }

    int32_t ns_dut=0, ns_ref=0;
    nms_onnx(boxes,scores,dut_out, 1,1,6, 3, bf16(0.5f),bf16(-1e9f), 0, &ns_dut);
    ns_ref=ref_nms(boxes_ref,scores,1,1,6,3,bf16(0.5f),bf16(-1e9f),ref_out);

    bool ok=triplet_match(dut_out,ns_dut,ref_out,ns_ref);
    printf("test_corner: num_selected dut=%d ref=%d %s\n",ns_dut,ns_ref,ok?"PASS":"FAIL");
    if(!ok){
        printf("  DUT:"); for(int i=0;i<ns_dut;i++) printf(" [%d,%d,%d]",dut_out[i*3],dut_out[i*3+1],dut_out[i*3+2]); printf("\n");
        printf("  REF:"); for(int i=0;i<ns_ref;i++) printf(" [%d,%d,%d]",ref_out[i*3],ref_out[i*3+1],ref_out[i*3+2]); printf("\n");
    }
    return ok;
}

// ── test case 2: center format, score_threshold filter ───────────────────────
static bool test_center_format()
{
    static bf16 boxes[1*MAX_BOXES*4]={};
    static bf16 scores[1*NUM_CLASSES*MAX_BOXES]={};
    static int32_t dut_out[MAX_SELECTED*3]={};
    static int32_t ref_out[MAX_SELECTED*3]={};

    // 4 center-format boxes [cx,cy,w,h] → corners for ref
    float raw_center[4][4]={{5,5,10,10},{5.5f,5.5f,10,10},{50,50,10,10},{51,51,10,10}};
    float sc[4]={0.9f,0.8f,0.7f,0.2f};  // box 3 below score_thr=0.25
    // Convert to FP32 corners after quantizing inputs through BF16, mirroring
    // the DUT's BF16-to-float load followed by center_to_corner().
    static coord_t boxes_corner[1*MAX_BOXES*4]={};
    for(int i=0;i<4;i++){
        coord_t cx=coord_t((float)bf16(raw_center[i][0]));
        coord_t cy=coord_t((float)bf16(raw_center[i][1]));
        coord_t w =coord_t((float)bf16(raw_center[i][2]));
        coord_t h =coord_t((float)bf16(raw_center[i][3]));
        coord_t hw=w*0.5f, hh=h*0.5f;
        boxes_corner[i*4+0]=cy-hh; boxes_corner[i*4+1]=cx-hw;
        boxes_corner[i*4+2]=cy+hh; boxes_corner[i*4+3]=cx+hw;
        scores[i]=bf16(sc[i]);
    }
    // DUT gets center format
    for(int i=0;i<4;i++)
        for(int k=0;k<4;k++) boxes[i*4+k]=bf16(raw_center[i][k]);

    int32_t ns_dut=0, ns_ref=0;
    nms_onnx(boxes,scores,dut_out, 1,1,4, 3, bf16(0.5f),bf16(0.25f), 1, &ns_dut);
    ns_ref=ref_nms(boxes_corner,scores,1,1,4,3,bf16(0.5f),bf16(0.25f),ref_out);

    bool ok=triplet_match(dut_out,ns_dut,ref_out,ns_ref);
    printf("test_center_format: num_selected dut=%d ref=%d %s\n",ns_dut,ns_ref,ok?"PASS":"FAIL");
    return ok;
}

// ── test case 3: large input, 1 batch, 1 class, MAX_BOXES boxes ─────────────
// Deterministic grid of mostly-non-overlapping boxes to exercise the O(N)
// radix sort and worst-case O(N^2) greedy NMS path at full capacity.
static bool test_large()
{
    static bf16 boxes[1*MAX_BOXES*4]={};
    static coord_t boxes_ref[1*MAX_BOXES*4]={};
    static bf16 scores[1*NUM_CLASSES*MAX_BOXES]={};
    static int32_t dut_out[MAX_SELECTED*3]={};
    static int32_t ref_out[MAX_SELECTED*3]={};

    const int N = MAX_BOXES;
    for(int i=0;i<N;i++){
        // Lay boxes on a stride-20 grid, size 10 → adjacent boxes don't overlap.
        float y = (float)((i % 64) * 20);
        float x = (float)((i / 64) * 20);
        boxes[i*4+0]=bf16(y);      boxes[i*4+1]=bf16(x);
        boxes[i*4+2]=bf16(y+10.f); boxes[i*4+3]=bf16(x+10.f);
        boxes_ref[i*4+0]=coord_t((float)bf16(y));      boxes_ref[i*4+1]=coord_t((float)bf16(x));
        boxes_ref[i*4+2]=coord_t((float)bf16(y+10.f)); boxes_ref[i*4+3]=coord_t((float)bf16(x+10.f));
        // Distinct descending-ish scores so the sort does real work.
        scores[i]=bf16(1.0f - (float)i*(1.0f/(float)N));
    }

    int32_t ns_dut=0, ns_ref=0;
    nms_onnx(boxes,scores,dut_out, 1,1,N, 0, bf16(0.5f),bf16(-1e9f), 0, &ns_dut);
    ns_ref=ref_nms(boxes_ref,scores,1,1,N,0,bf16(0.5f),bf16(-1e9f),ref_out);

    bool ok=triplet_match(dut_out,ns_dut,ref_out,ns_ref);
    printf("test_large: num_selected dut=%d ref=%d %s\n",ns_dut,ns_ref,ok?"PASS":"FAIL");
    return ok;
}

// ── test case 4: 2 classes, 6300 boxes ──────────────────────────────────────
// Exercises NUM_CLASSES=2 with num_boxes=6300 (requires MAX_BOXES>=6300).
static bool test_two_class_6300()
{
    const int nb = 1, nc = 2, nx = 6300;
    if (nx > MAX_BOXES || nc > NUM_CLASSES) {
        printf("test_two_class_6300: SKIP (needs MAX_BOXES>=6300, NUM_CLASSES>=2; "
               "built with MAX_BOXES=%d NUM_CLASSES=%d)\n", MAX_BOXES, NUM_CLASSES);
        return true;
    }

    static bf16    boxes [1*MAX_BOXES*4]           = {};
    static coord_t boxes_ref[1*MAX_BOXES*4]        = {};
    static bf16    scores[1*NUM_CLASSES*MAX_BOXES] = {};
    static int32_t dut_out[MAX_SELECTED*3]         = {};
    static int32_t ref_out[MAX_SELECTED*3]         = {};

    // Grid of boxes: stride 20, size 30. Neighbours overlap, but their IoU is
    // below 0.5, so this case selects all 6300 boxes in each class.
    for (int i = 0; i < nx; i++) {
        float y = (float)((i % 80) * 20);
        float x = (float)((i / 80) * 20);
        boxes[i*4+0] = bf16(y);        boxes[i*4+1] = bf16(x);
        boxes[i*4+2] = bf16(y + 30.f); boxes[i*4+3] = bf16(x + 30.f);
        boxes_ref[i*4+0] = coord_t((float)bf16(y));        boxes_ref[i*4+1] = coord_t((float)bf16(x));
        boxes_ref[i*4+2] = coord_t((float)bf16(y + 30.f)); boxes_ref[i*4+3] = coord_t((float)bf16(x + 30.f));
    }
    // Distinct per-class scores so each class sorts/suppresses differently.
    for (int c = 0; c < nc; c++)
        for (int n = 0; n < nx; n++)
            scores[(0*NUM_CLASSES + c)*MAX_BOXES + n] =
                bf16(1.0f - (float)((n + c*7) % nx) * (1.0f / (float)nx));

    int32_t ns_dut = 0, ns_ref = 0;
    nms_onnx(boxes, scores, dut_out, nb, nc, nx, 0, bf16(0.5f), bf16(-1e9f), 0, &ns_dut);
    ns_ref = ref_nms(boxes_ref, scores, nb, nc, nx, 0, bf16(0.5f), bf16(-1e9f), ref_out);

    bool ok = triplet_match(dut_out, ns_dut, ref_out, ns_ref);
    printf("test_two_class_6300: num_selected dut=%d ref=%d %s\n", ns_dut, ns_ref, ok ? "PASS" : "FAIL");
    return ok;
}

// ── test case 5: Case B — 2 classes, 6300 boxes, heavy suppression ───────────
// All 6300 boxes clear the score threshold (num_cand=6300, so the sort and the
// outer greedy loop run at full size), but the boxes form ~100 tight clusters of
// identical boxes. Each cluster collapses to a single selection, so only ~100
// boxes survive per class (~200 total). The inner scan still traverses the
// remaining candidate ranks for every accepted cluster representative, while
// many individual IoU calculations are skipped for already-suppressed ranks.
static bool test_two_class_6300_caseB()
{
    const int nb = 1, nc = 2, nx = 6300;
    if (nx > MAX_BOXES || nc > NUM_CLASSES) {
        printf("test_two_class_6300_caseB: SKIP (needs MAX_BOXES>=6300, NUM_CLASSES>=2; "
               "built with MAX_BOXES=%d NUM_CLASSES=%d)\n", MAX_BOXES, NUM_CLASSES);
        return true;
    }

    static bf16    boxes [1*MAX_BOXES*4]           = {};
    static coord_t boxes_ref[1*MAX_BOXES*4]        = {};
    static bf16    scores[1*NUM_CLASSES*MAX_BOXES] = {};
    static int32_t dut_out[MAX_SELECTED*3]         = {};
    static int32_t ref_out[MAX_SELECTED*3]         = {};

    // 100 clusters × 63 boxes = 6300. Clusters sit on a 10×10 grid at stride 100
    // (box size 30 → ~70px gap, no cross-cluster overlap). Within a cluster all
    // boxes are identical → mutual IoU=1.0 > 0.5, so each cluster keeps exactly
    // one box. Result: ~100 selected per class.
    const int CLUSTERS = 100, PER = 63;   // 100*63 = 6300 = nx
    for (int i = 0; i < nx; i++) {
        int cidx = i / PER;               // 0..99
        int gx = cidx % 10, gy = cidx / 10;
        float X = (float)(gx * 100), Y = (float)(gy * 100);
        boxes[i*4+0] = bf16(Y);        boxes[i*4+1] = bf16(X);
        boxes[i*4+2] = bf16(Y + 30.f); boxes[i*4+3] = bf16(X + 30.f);
        boxes_ref[i*4+0] = coord_t((float)bf16(Y));        boxes_ref[i*4+1] = coord_t((float)bf16(X));
        boxes_ref[i*4+2] = coord_t((float)bf16(Y + 30.f)); boxes_ref[i*4+3] = coord_t((float)bf16(X + 30.f));
    }
    // Distinct-ish per-class scores; ties within a cluster are broken stably
    // (lowest index) by both the DUT radix and the reference insertion sort.
    for (int c = 0; c < nc; c++)
        for (int n = 0; n < nx; n++)
            scores[(0*NUM_CLASSES + c)*MAX_BOXES + n] =
                bf16(1.0f - (float)((n + c*7) % nx) * (1.0f / (float)nx));

    int32_t ns_dut = 0, ns_ref = 0;
    nms_onnx(boxes, scores, dut_out, nb, nc, nx, 0, bf16(0.5f), bf16(-1e9f), 0, &ns_dut);
    ns_ref = ref_nms(boxes_ref, scores, nb, nc, nx, 0, bf16(0.5f), bf16(-1e9f), ref_out);

    bool ok = triplet_match(dut_out, ns_dut, ref_out, ns_ref);
    printf("test_two_class_6300_caseB: num_selected dut=%d ref=%d %s\n", ns_dut, ns_ref, ok ? "PASS" : "FAIL");
    return ok;
}

// ── test case 6: Case A — 2 classes, 6300 boxes, hard score filter ───────────
// All 6300 boxes are laid out non-overlapping (nothing ever suppresses), but the
// score threshold admits only ~100 boxes per class. Here num_cand≈100, so the
// radix sort and the outer greedy loop shrink to ~100 as well — the opposite of
// Case B, where those stay maximal. Every surviving candidate is selected, so
// num_selected≈200 total. DMA load still transfers all 6300 boxes.
static bool test_two_class_6300_caseA()
{
    const int nb = 1, nc = 2, nx = 6300;
    if (nx > MAX_BOXES || nc > NUM_CLASSES) {
        printf("test_two_class_6300_caseA: SKIP (needs MAX_BOXES>=6300, NUM_CLASSES>=2; "
               "built with MAX_BOXES=%d NUM_CLASSES=%d)\n", MAX_BOXES, NUM_CLASSES);
        return true;
    }

    static bf16    boxes [1*MAX_BOXES*4]           = {};
    static coord_t boxes_ref[1*MAX_BOXES*4]        = {};
    static bf16    scores[1*NUM_CLASSES*MAX_BOXES] = {};
    static int32_t dut_out[MAX_SELECTED*3]         = {};
    static int32_t ref_out[MAX_SELECTED*3]         = {};

    // Non-overlapping grid: stride 20, size 10 → no box ever suppresses another.
    for (int i = 0; i < nx; i++) {
        float y = (float)((i % 64) * 20);
        float x = (float)((i / 64) * 20);
        boxes[i*4+0] = bf16(y);        boxes[i*4+1] = bf16(x);
        boxes[i*4+2] = bf16(y + 10.f); boxes[i*4+3] = bf16(x + 10.f);
        boxes_ref[i*4+0] = coord_t((float)bf16(y));        boxes_ref[i*4+1] = coord_t((float)bf16(x));
        boxes_ref[i*4+2] = coord_t((float)bf16(y + 10.f)); boxes_ref[i*4+3] = coord_t((float)bf16(x + 10.f));
    }
    // Score filter: 100 boxes per class score above threshold, the rest at 0.
    // Class 0 admits indices [0,100), class 1 admits a disjoint [100,200) so the
    // two classes select different boxes. Passing scores descend distinctly to
    // give the radix sort real work; all stay well above the 0.25 threshold.
    const int PASS = 100;
    for (int c = 0; c < nc; c++) {
        int lo = c * PASS;            // class 0 → [0,100), class 1 → [100,200)
        for (int n = 0; n < nx; n++) {
            float s = 0.0f;
            if (n >= lo && n < lo + PASS)
                s = 1.0f - (float)(n - lo) * (0.5f / (float)PASS);  // 1.0 → 0.505
            scores[(0*NUM_CLASSES + c)*MAX_BOXES + n] = bf16(s);
        }
    }

    int32_t ns_dut = 0, ns_ref = 0;
    nms_onnx(boxes, scores, dut_out, nb, nc, nx, 0, bf16(0.5f), bf16(0.25f), 0, &ns_dut);
    ns_ref = ref_nms(boxes_ref, scores, nb, nc, nx, 0, bf16(0.5f), bf16(0.25f), ref_out);

    bool ok = triplet_match(dut_out, ns_dut, ref_out, ns_ref);
    printf("test_two_class_6300_caseA: num_selected dut=%d ref=%d %s\n", ns_dut, ns_ref, ok ? "PASS" : "FAIL");
    return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main()
{
    bool all_pass = true;
    all_pass &= test_corner();
    all_pass &= test_center_format();
    all_pass &= test_two_class_6300_caseA();
    all_pass &= test_two_class_6300_caseB();
//    all_pass &= test_two_class_6300();

    printf("\nOverall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
