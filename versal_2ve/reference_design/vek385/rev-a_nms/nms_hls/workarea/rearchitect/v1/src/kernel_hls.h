#ifndef KERNEL_HLS_H
#define KERNEL_HLS_H

#include <ap_float.h>
#include <stdint.h>

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

void nms_onnx(
    const bf16_t *boxes_in,
    const bf16_t *scores_in,
    int32_t      *selected_out,
    int           num_batches,
    int           num_classes,
    int           num_boxes,
    int           max_out_per_class,
    bf16_t        iou_threshold,
    bf16_t        score_threshold,
    int           center_point_box,
    int32_t      *num_selected);

#endif
