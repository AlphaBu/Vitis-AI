#include <cstdint>
#include "ap_int.h"

// 32-bit memory-mapped data word.
typedef ap_uint<32> data_t;

extern "C" {
void pass_through(const data_t *in,    // input  vector in global memory (32-bit)
                  data_t       *out,   // output vector in global memory (32-bit)
                  int           size)  // number of 32-bit words to copy
{
    // Data movers to global memory (DDR/LPDDR). 32-bit AXI4 data width.
#pragma HLS INTERFACE m_axi     port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi     port=out bundle=gmem1 offset=slave
    // All s_axilite ports must share ONE bundle in Vitis kernel mode:
    // map each pointer's offset register + scalar + return to 'control'.
#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = in[i];   // forward input word to output (burst-inferable)
    }
}
}
