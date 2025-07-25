#ifndef FC_PARAMS_H
#define FC_PARAMS_H

#define FC_INPUT_SIZE 256
#define FC_OUTPUT_SIZE 20

extern const float fc_weights[FC_OUTPUT_SIZE * FC_INPUT_SIZE]; // using flattened 1d array
extern const float fc_biases[FC_OUTPUT_SIZE];

#endif
