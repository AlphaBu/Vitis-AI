#include <cstdio>
#include "ap_int.h"

// 512-bit memory-mapped data word (one AXI4 beat = 64 bytes).
typedef ap_uint<512> data_t;

extern "C" void pass_through(const data_t *in, data_t *out, int size);

int main() {
    const int SIZE = 64;          // number of 512-bit beats
    data_t in[SIZE];
    data_t out[SIZE];

    // Drive the input buffer with a known pattern: pack 16 distinct 32-bit
    // sub-words into each 512-bit beat so every byte lane is exercised.
    for (int i = 0; i < SIZE; i++) {
        data_t v = 0;
        for (int w = 0; w < 16; w++) {
            v.range(32 * w + 31, 32 * w) =
                ap_uint<32>(0xA0000000u + (unsigned)(i * 16 + w));
        }
        in[i]  = v;
        out[i] = 0;
    }

    pass_through(in, out, SIZE);

    // Check the output matches the input beat-for-beat.
    int errors = 0;
    for (int i = 0; i < SIZE; i++) {
        if (out[i] != in[i]) {
            printf("ERROR: beat %d mismatch\n", i);
            errors++;
        }
    }

    if (errors == 0) printf("PASS: %d beats (512-bit) forwarded correctly\n", SIZE);
    else             printf("FAIL: %d error(s)\n", errors);
    return errors;
}
