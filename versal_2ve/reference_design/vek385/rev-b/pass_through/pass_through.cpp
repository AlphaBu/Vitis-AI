#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"

// 512-bit memory-mapped data word (one AXI4 beat = 64 bytes).
typedef ap_uint<512> data_t;

// Internal FIFO depth: the on-chip buffer holds up to 4 x 512-bit words.
#define BUF_DEPTH 16

// ---------------------------------------------------------------------------
// Stage 1 (read): burst-read 512-bit words from global memory into the input
// buffer FIFO. Decouples DDR/LPDDR read latency from the compute stage.
// ---------------------------------------------------------------------------
static void read_input(const data_t *in, hls::stream<data_t> &in_buf, int size) {
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        in_buf.write(in[i]);   // burst-inferable sequential read
    }
}

// ---------------------------------------------------------------------------
// Stage 2 (process): read from the input buffer and pass each word through to
// the output buffer. This is where any per-word processing would live.
// ---------------------------------------------------------------------------
static void process(hls::stream<data_t> &in_buf,
                    hls::stream<data_t> &out_buf, int size) {
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out_buf.write(in_buf.read());   // pass-through
    }
}

// ---------------------------------------------------------------------------
// Stage 3 (write): drain the output buffer FIFO to global memory.
// ---------------------------------------------------------------------------
static void write_output(data_t *out, hls::stream<data_t> &out_buf, int size) {
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = out_buf.read();   // burst-inferable sequential write
    }
}

extern "C" {
void pass_through(const data_t *in,    // input  vector in global memory (512-bit)
                  data_t       *out,   // output vector in global memory (512-bit)
                  int           size)  // number of 512-bit words (beats) to copy
{
    // Data movers to global memory (DDR/LPDDR). 512-bit AXI4 data width
    // (derived from the ap_uint<512> element type). 'depth' is a C/RTL
    // co-simulation modelling hint only (matches the testbench beat count);
    // it does not affect the synthesized hardware or runtime behavior.
#pragma HLS INTERFACE m_axi     port=in  bundle=gmem0 offset=slave depth=64
#pragma HLS INTERFACE m_axi     port=out bundle=gmem1 offset=slave depth=64
    // All s_axilite ports must share ONE bundle in Vitis kernel mode:
    // map each pointer's offset register + scalar + return to 'control'.
#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Run read / process / write as concurrent tasks (task-level pipelining),
    // connected by 4-deep 512-bit on-chip FIFO buffers.
#pragma HLS DATAFLOW
    hls::stream<data_t> in_buf;
#pragma HLS STREAM variable=in_buf  depth=BUF_DEPTH
    hls::stream<data_t> out_buf;
#pragma HLS STREAM variable=out_buf depth=BUF_DEPTH

    read_input (in,  in_buf,  size);   // stage 1: cache input into buffer
    process    (in_buf, out_buf, size);// stage 2: read buffer -> pass through
    write_output(out, out_buf, size);  // stage 3: output buffer -> global mem
}
}
