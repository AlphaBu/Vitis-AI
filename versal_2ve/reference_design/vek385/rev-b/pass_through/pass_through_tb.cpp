#include <cstdio>
#include "ap_int.h"

typedef ap_uint<32> data_t;

extern "C" void pass_through(const data_t *in, data_t *out, int size);

int main() {
    const int SIZE = 64;
    data_t in[SIZE];
    data_t out[SIZE];

    // Drive the input buffer with a known pattern.
    for (int i = 0; i < SIZE; i++) {
        in[i]  = 0xA0000000u + i;
        out[i] = 0;
    }

    pass_through(in, out, SIZE);

    // Check the output matches the input word-for-word.
    int errors = 0;
    for (int i = 0; i < SIZE; i++) {
        if (out[i] != in[i]) {
            printf("ERROR: idx %d got 0x%08X exp 0x%08X\n",
                   i, (unsigned)out[i], (unsigned)in[i]);
            errors++;
        }
    }

    if (errors == 0) printf("PASS: %d words forwarded correctly\n", SIZE);
    else             printf("FAIL: %d error(s)\n", errors);
    return errors;
}
