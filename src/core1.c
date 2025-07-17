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

static float nn_input[FC_INPUT_SIZE];
static float nn_output[FC_OUTPUT_SIZE];

// -----------------------------------------------------------------------------
// Functions
// -----------------------------------------------------------------------------

// converts 8-bit samples (MSB of each word) to float and normalize
void load_input(uint32_t *raw_buffer) { 
    for (int i = 0; i < FC_INPUT_SIZE; i++) { 
        uint8_t raw_sample = (raw_buffer[i] >> 24) & 0xFF; 
        nn_input[i] = (float)raw_sample / 255.0f; 
    }
}

// runs single-layer linear neural network 
void run_inference(uint32_t *buf) { 
    load_input(buf); 
    lin(nn_input, fc_weights, fc_biases, nn_output, FC_INPUT_SIZE, FC_OUTPUT_SIZE); 
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) { 
        printf("Output[%d] = %.2f\n", i, nn_output[i]); 
    }
}

// -----------------------------------------------------------------------------
// Core1_Main
// -----------------------------------------------------------------------------

void core1_main(void) {
    stdio_init_all();
    multicore_fifo_push_blocking(0xABBAABBA); // push signal word to core0

    while (true) {
        uintptr_t raw = multicore_fifo_pop_blocking();
        float *mag = (float*)raw;
    
        lin(mag, fc_weights, fc_biases, nn_output, FC_INPUT_SIZE, FC_OUTPUT_SIZE);

        static uint32_t frame_no = 0;
        if (++frame_no >= 50) {         
            frame_no = 0;
           for (int i = 0; i < FC_OUTPUT_SIZE; ++i)
                printf("%2d:% .2f ", i, nn_output[i]);
            printf("\n");
        }
        
        multicore_fifo_push_blocking(raw);
      
    }
}

