#include "operators.h"
#include <stddef.h>
#include <stdint.h>
#include <stddef.h>
#include "fc_params.h"

void lin(const sample_t *in,
         const sample_t *W,
         const sample_t *b,
         sample_t       *out,
         size_t          in_dim,
         size_t          out_dim) {
    for (size_t j = 0; j < out_dim; j++) {
        sample_t acc = b[j];
        for (size_t i = 0; i < in_dim; i++) {
            acc += W[j * in_dim + i] * in[i];
        }
        out[j] = acc;
    }
}

