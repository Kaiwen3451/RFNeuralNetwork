#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"

#include "fc_params.h"     
#include "operators.h"   

// -----------------------------------------------------------------------------
// Globals
// ----------------------------------------------------------------------------- 

#define FEAT_DIM 6 
static float nn_input[FEAT_DIM];
static float nn_output[FC_OUTPUT_SIZE];

typedef struct {
    float centroid;
    float bandwidth;
    float entropy;
    float peak1; 
    float peak2;
    float peak3;
} features_t;

// -----------------------------------------------------------------------------
// Functions (legacy)
// -----------------------------------------------------------------------------

// // converts 8-bit samples (MSB of each word) to float and normalize
// void load_input(uint32_t *raw_buffer) { 
//     for (int i = 0; i < FC_INPUT_SIZE; i++) { 
//         uint8_t raw_sample = (raw_buffer[i] >> 24) & 0xFF; 
//         nn_input[i] = (float)raw_sample / 255.0f; 
//     }
// }

// // runs single-layer linear neural network 
// void run_inference(uint32_t *buf) { 
//     load_input(buf); 
//     lin(nn_input, fc_weights, fc_biases, nn_output, FC_INPUT_SIZE, FC_OUTPUT_SIZE); 
//     for (int i = 0; i < FC_OUTPUT_SIZE; i++) { 
//         printf("Output[%d] = %.2f\n", i, nn_output[i]); 
//     }
// }

// -----------------------------------------------------------------------------
// Core1_Main
// -----------------------------------------------------------------------------

void core1_main(void) {
    stdio_init_all();
    multicore_fifo_push_blocking(0xABBAABBA); // push signal word to core0

    while (true) {
        uintptr_t raw = multicore_fifo_pop_blocking();
        features_t *f = (features_t *)raw;

        nn_input[0] = f->centroid;
        nn_input[1] = f->bandwidth;
        nn_input[2] = f->entropy;
        nn_input[3] = f->peak1;
        nn_input[4] = f->peak2;
        nn_input[5] = f->peak3;

        lin(nn_input, fc_weights, fc_biases, nn_output,
            FEAT_DIM, FC_OUTPUT_SIZE);

        static uint32_t x = 0; 
        if (++x >= 512 ) {         
            x = 0;
            for (int i = 0; i < FC_OUTPUT_SIZE; ++i)
                printf("%2d:% .4f ", i, nn_output[i]);
            printf("\n");
        }
        multicore_fifo_push_blocking(raw);   
    }


    
}

