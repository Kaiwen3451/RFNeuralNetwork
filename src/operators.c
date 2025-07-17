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

void conv(const sample_t *in,
          const sample_t *ker,
          sample_t       *out,
          size_t          i_rows,
          size_t          i_cols,
          size_t          k_rows,
          size_t          k_cols) {

    ptrdiff_t p_row = (ptrdiff_t)(k_rows - 1) / 2;
    ptrdiff_t p_col = (ptrdiff_t)(k_cols - 1) / 2;

    for (size_t i = 0; i < i_rows; i++) {
        for (size_t j = 0; j < i_cols; j++) {
            sample_t acc = 0;
            for (size_t ki = 0; ki < k_rows; ki++) {
                for (size_t kj = 0; kj < k_cols; kj++) {
                    ptrdiff_t in_r = (ptrdiff_t)i + (ptrdiff_t)ki - p_row;
                    ptrdiff_t in_c = (ptrdiff_t)j + (ptrdiff_t)kj - p_col;
                    if (in_r >= 0 && in_r < (ptrdiff_t)i_rows &&
                        in_c >= 0 && in_c < (ptrdiff_t)i_cols) {
                        acc += in[(size_t)in_r * i_cols + (size_t)in_c]
                             * ker[ki * k_cols + kj];
                    }
                }
            }
            out[i * i_cols + j] = acc;
        }
    }
}

