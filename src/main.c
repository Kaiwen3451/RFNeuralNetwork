#include <stddef.h>  
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/vreg.h"
#include "hardware/clocks.h"

#include "bit_rx.pio.h"
#include "fc_params.h"

// -----------------------------------------------------------------------------
// Definitions
// -----------------------------------------------------------------------------

#define DATA_BASE_PIN  0 // assume ADC has 8-bit res
#define DATA_VALID     8 // at ADC's output speed

#define FFT_SIZE       512 
#define FFT_HALF       (FFT_SIZE/2) // do these at compile time for the FFT
#define LOG2_FFT       9          
#define SHIFT          (16 - LOG2_FFT)

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

// ping-pong buffers using 32-bit (FIFO width) words
// size determined by how many points we are taking per FFT  
static uint32_t buffer_a[FFT_SIZE] __attribute__((aligned(4)));   
static uint32_t buffer_b[FFT_SIZE] __attribute__((aligned(4))); 
static bool use_a = true;

// holds the DMA channel index returned by dma_claim_unused_channel()
static int dma_chan;

// sin lookup table
extern const float Sinewave[FFT_SIZE];

// windowing coefficients 
static float hann[FFT_SIZE];

// real and imag part of frequency bins after fft_float
// passed into the NN as a magnitude, and since all FFT inputs are real we take half of the spectrogram
static float fr[FFT_SIZE], fi[FFT_SIZE];
static float mag_buf[FFT_HALF];

// -----------------------------------------------------------------------------
// Helpers 
// -----------------------------------------------------------------------------

static void init_hann_window(void) {
    const float PI = 3.14159265358979323846f;
    for (int i = 0; i < FFT_SIZE; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2*PI*i/(FFT_SIZE-1)));
    }
}

// FFT adopted from https://vanhunteradams.com/FFT/FFT.html
static void bit_reverse(void) {
    for (uint16_t m = 1; m < FFT_SIZE-1; m++) {
        uint16_t mr = ((m>>1)&0x5555)|((m&0x5555)<<1);
        mr = ((mr>>2)&0x3333)|((mr&0x3333)<<2);
        mr = ((mr>>4)&0x0F0F)|((mr&0x0F0F)<<4);
        mr = ((mr>>8)&0x00FF)|((mr&0x00FF)<<8);
        mr >>= SHIFT;
        if (mr <= m) continue;
        float t = fr[m]; fr[m] = fr[mr]; fr[mr] = t;
        t    = fi[m]; fi[m] = fi[mr]; fi[mr] = t;
    }
}

static void fft_float(void) {
    bit_reverse();
    int half = 1, shift = LOG2_FFT-1;
    while (half < FFT_SIZE) {
        int step = half << 1;
        for (int m = 0; m < half; m++) {
            float wr =  0.5f *  Sinewave[m << shift];
            float wi = -0.5f * Sinewave[(m << shift) + FFT_SIZE/4];
            for (int i = m; i < FFT_SIZE; i += step) {
                int j = i + half;
                float tr = wr * fr[j] - wi * fi[j];
                float ti = wr * fi[j] + wi * fr[j];
                float qr = fr[i] * 0.5f;      
                float qi = fi[i] * 0.5f;
                fr[j] = qr - tr;   fi[j] = qi - ti;
                fr[i] = qr + tr;   fi[i] = qi + ti;
            }
        }
        half <<= 1;
        shift--;
    }
}

// -----------------------------------------------------------------------------
// DMA interrupt handler
// -----------------------------------------------------------------------------
void dma_handler() {
    dma_hw->ints0 = 1u << dma_chan; // clear interrupt flag

    uint32_t *src = use_a ? buffer_a : buffer_b;

    for (int i = 0; i < FFT_SIZE; i++) {
        uint8_t s = (src[i] >> 24) & 0xFF;
        float x = (s / 255.0f) - 0.5f; // normalize to [-1/2,1/2]
        fr[i] = x * hann[i]; // windowing it
        fi[i] = 0;
    }

    fft_float();

    for (int i = 0; i < FFT_HALF; i++) {
        mag_buf[i] = hypotf(fr[i], fi[i]);
    }

    if (multicore_fifo_wready()) {
        multicore_fifo_push_blocking((uintptr_t)mag_buf);
    } else {
    }

    uint32_t *next = use_a ? buffer_b : buffer_a;
    dma_channel_set_write_addr(dma_chan, next, true);
    use_a = !use_a;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
void main(void) {

    // overclocking to 378 MHz
    vreg_set_voltage(VREG_VOLTAGE_1_30);    
    sleep_ms(5);
    if (!set_sys_clock_khz(378000, true)) {
        printf("SYS CLK SPEED NOT ATTAINABLE\n");
        while (1) { __breakpoint();}
    }

    stdio_init_all(); 
    gpio_init(PICO_DEFAULT_LED_PIN);    
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);   
    gpio_put(PICO_DEFAULT_LED_PIN, 1);  // program is on     
    init_hann_window();

    /* PIO Setup */ 
    // Grabs 8 input pins in parallel on rising CLK_PIN and pushes a 32-bit word
    // Most-significant byte in the word is the input values
    PIO  pio    = pio0;     
    uint offset = pio_add_program(pio, &bit_rx_program);    
    uint sm     = pio_claim_unused_sm(pio, true);   

    for (int i = 0; i < 8; ++i) {
        pio_gpio_init(pio, DATA_BASE_PIN + i);
    } 
    pio_gpio_init(pio, DATA_VALID); 

    pio_sm_config c = bit_rx_program_get_default_config(offset);    
    sm_config_set_in_pins(&c, DATA_BASE_PIN);   
    sm_config_set_in_shift(&c, true, true, 8); // autopushes every 8 bits, saves an PIO inst
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);  // joins so the RX can hold 8 entries
    sm_config_set_clkdiv_int_frac(&c, 1, 0);    

    pio_sm_init(pio, sm, offset, &c);   
    pio_sm_clear_fifos(pio, sm);
    pio_sm_set_enabled(pio, sm, true);  

    /* DMA Setup */
    // Streams data from the PIO FIFO into the RAM 
    dma_chan = dma_claim_unused_channel(true);  
    dma_channel_config d = dma_channel_get_default_config(dma_chan);    

    channel_config_set_read_increment (&d, false);  // read from PIO FIFO always
    channel_config_set_write_increment(&d, true);   
    channel_config_set_transfer_data_size(&d, DMA_SIZE_32); // matching FIFO width          
    channel_config_set_dreq(&d, pio_get_dreq(pio, sm, false));  

    dma_channel_configure(dma_chan, &d,
        buffer_a,           // *write_addr
        &pio->rxf[sm],      // *read_addr
        FFT_SIZE,           // transfer_count
        false               // trigger
    );

    dma_channel_set_irq0_enabled(dma_chan, true); // raise interrupt when done
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);

    dma_channel_start(dma_chan);
    
    /* Running Inference */
    // Inference occurs on core1
    extern void core1_main(void);                 
    multicore_launch_core1(core1_main);           
    multicore_fifo_pop_blocking();           

    while (true) {
        (void)multicore_fifo_pop_blocking(); // wait until core 1 finishes its inference
        tight_loop_contents(); 
    }
} 