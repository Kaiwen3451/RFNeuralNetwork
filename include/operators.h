#ifndef OPERATORS_H
#define OPERATORS_H

#include <stdint.h>
#include <stddef.h>

typedef float sample_t;

void lin(const sample_t *in,
         const sample_t *W,
         const sample_t *b,
         sample_t       *out,
         size_t          in_dim,
         size_t          out_dim);

#endif 
