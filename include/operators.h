#ifndef OPERATORS_H
#define OPERATORS_H

#include <stdint.h>
#include <stddef.h>

typedef float sample_t;

/**
 * Fully-connected linear layer
 *
 * Computes out = W * in + b
 *
 * @param in_dim  Dimension of input
 * @param out_dim Dimension of output
 */
void lin(const sample_t *in,
         const sample_t *W,
         const sample_t *b,
         sample_t       *out,
         size_t          in_dim,
         size_t          out_dim);

/**
 * @brief 2D convolution
 *
 * Convolves a kernel K (k_rows x k_cols) over an input I (i_rows x i_cols),
 * producing an output O (i_rows x i_cols) with zero-padding.
 *
 * @param in     Input matrix, 
 * @param ker    Kernel matrix
 * @param out    Output matrix
 * @param i_rows Number of rows in input
 * @param i_cols Number of cols in input
 * @param k_rows Number of rows in kernel
 * @param k_cols Number of cols in kernel
 */
void conv(const sample_t *in,
          const sample_t *ker,
          sample_t       *out,
          size_t          i_rows,
          size_t          i_cols,
          size_t          k_rows,
          size_t          k_cols);

#endif // OPERATORS_H
