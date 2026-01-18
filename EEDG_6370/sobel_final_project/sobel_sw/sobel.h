#include "hls_task.h"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#define PIXEL_LOOP 3
#define ROW_LOOP 3
#define MEMWIDTH 8
#define DATAWIDTH 32
#define ADDRWIDTH 11
#define ELEMENTS 1<<ADDRWIDTH

typedef ap_uint<MEMWIDTH> mem_t;
typedef ap_uint<DATAWIDTH> conc_t;
typedef ap_uint<MEMWIDTH * 2> l_mem_t;
typedef ap_axiu<DATAWIDTH, 0, 0, 0> sob_axi_str_t;
typedef struct {
  // mem_t blank;
  mem_t r;
  mem_t g;
  mem_t b;
} data_comp_t;
typedef struct {
  // mem_t blank;
  int r;
  int g;
  int b;
} l_data_comp_t;
typedef ap_uint<DATAWIDTH> data_t;


void top_sobel( hls::stream<sob_axi_str_t> &in, hls::stream<sob_axi_str_t> &out);
