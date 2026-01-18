#include "sobel.h"
#include <hls_task.h>

void top_sobel(hls::stream<sob_axi_str_t> &data_in, hls::stream<sob_axi_str_t> &data_out)
{
  #pragma HLS INTERFACE mode=ap_ctrl_none port=return
  #pragma HLS INTERFACE mode=axis register_mode=both port=data_out
  #pragma HLS INTERFACE mode=axis register_mode=both port=data_in
  mem_t mem_red[ELEMENTS][ROW_LOOP]   ;
  mem_t mem_green[ELEMENTS][ROW_LOOP] ;
  mem_t mem_blue[ELEMENTS][ROW_LOOP]  ;
  #pragma HLS BIND_STORAGE variable=mem_red type=ram_1p impl=bram
  #pragma HLS BIND_STORAGE variable=mem_green type=ram_1p impl=bram
  #pragma HLS BIND_STORAGE variable=mem_blue type=ram_1p impl=bram
  #pragma HLS ARRAY_PARTITION variable=mem_red dim=2 factor=3 type=block
  #pragma HLS ARRAY_PARTITION variable=mem_green dim=2 factor=3 type=block
  #pragma HLS ARRAY_PARTITION variable=mem_blue dim=2 factor=3 type=block

  int wptr_l;
  bool wptr_done = false;
  static int offset = 0;
  sob_axi_str_t tmp;
  WRITE_LOOP: 
  for (int wptr = 1; wptr < ELEMENTS; wptr++) {
    #pragma HLS PIPELINE II=1
    if (data_in.empty()==0) {
      tmp = data_in.read();
      mem_red[wptr][offset].write(tmp.data.range(3 * MEMWIDTH - 1, 2 * MEMWIDTH));
      mem_green[wptr][offset].write(tmp.data.range(2 * MEMWIDTH - 1, 1 * MEMWIDTH));
      mem_blue[wptr][offset].write(tmp.data.range(1 * MEMWIDTH - 1, 0 * MEMWIDTH));
    }
    wptr_l++;
    // wptr_done = false;
    if (tmp.last == 1) {
      offset = (offset + 1) % ROW_LOOP;
      wptr_l = wptr;
      wptr_done = true;
      break;
    }
  }

  sob_axi_str_t read_tmp;
  l_mem_t conc_temp;
  l_data_comp_t sum, sumT, sumX, sumY;
  data_comp_t shift_reg_0[3], shift_reg_1[3], shift_reg_2[3], sumF;
  READ_LOOP: 
  for (int rptr = 0; rptr < ELEMENTS; rptr++) {
    #pragma HLS PIPELINE II=1

    shift_reg_2[2] = shift_reg_1[2];
    shift_reg_1[2] = shift_reg_0[2];
    shift_reg_0[2].r = mem_red[rptr][(offset + 2) % ROW_LOOP];
    shift_reg_0[2].g = mem_green[rptr][(offset + 2) % ROW_LOOP];
    shift_reg_0[2].b = mem_blue[rptr][(offset + 2) % ROW_LOOP];

    shift_reg_2[1] = shift_reg_1[1];
    shift_reg_1[1] = shift_reg_0[1];
    shift_reg_0[1].r = mem_red[rptr][(offset + 1) % ROW_LOOP];
    shift_reg_0[1].g = mem_green[rptr][(offset + 1) % ROW_LOOP];
    shift_reg_0[1].b = mem_blue[rptr][(offset + 1) % ROW_LOOP];

    shift_reg_2[0] = shift_reg_1[2];
    shift_reg_1[0] = shift_reg_0[2];
    shift_reg_0[0].r = mem_red[rptr][(offset + 0) % ROW_LOOP];
    shift_reg_0[0].g = mem_green[rptr][(offset + 0) % ROW_LOOP];
    shift_reg_0[0].b = mem_blue[rptr][(offset + 0) % ROW_LOOP];

    sumY.r = shift_reg_0[0].r + 2 *  shift_reg_0[1].r + shift_reg_0[2].r - (shift_reg_2[0].r + 2 *  shift_reg_2[1].r + shift_reg_2[2].r);
    sumY.g = shift_reg_0[0].g + 2 *  shift_reg_0[1].g + shift_reg_0[2].g - (shift_reg_2[0].g + 2 *  shift_reg_2[1].g + shift_reg_2[2].g);
    sumY.b = shift_reg_0[0].b + 2 *  shift_reg_0[1].b + shift_reg_0[2].b - (shift_reg_2[0].b + 2 *  shift_reg_2[1].b + shift_reg_2[2].b);

    sumX.r = shift_reg_0[0].r + 2 *  shift_reg_1[0].r + shift_reg_2[0].r - (shift_reg_0[2].r + 2 *  shift_reg_1[2].r + shift_reg_2[2].r);
    sumX.g = shift_reg_0[0].g + 2 *  shift_reg_1[0].g + shift_reg_2[0].g - (shift_reg_0[2].g + 2 *  shift_reg_1[2].g + shift_reg_2[2].g);
    sumX.b = shift_reg_0[0].b + 2 *  shift_reg_1[0].b + shift_reg_2[0].b - (shift_reg_0[2].b + 2 *  shift_reg_1[2].b + shift_reg_2[2].b);

    sumT.r = sumX.r + sumY.r;
    sumT.g = sumX.g + sumY.g;
    sumT.b = sumX.b + sumY.b;

    sum.r = (sumT.r < 0) ? 0 : sumT.r; 
    sum.g = (sumT.g < 0) ? 0 : sumT.r; 
    sum.b = (sumT.b < 0) ? 0 : sumT.r; 

    sumF.r = (sum.r > 255) ? 255 : sum.r; 
    sumF.g = (sum.g > 255) ? 255 : sum.r; 
    sumF.b = (sum.b > 255) ? 255 : sum.r; 

    conc_temp = sumF.g.concat(sumF.b);
    read_tmp.data = sumF.r.concat(conc_temp);
    read_tmp.keep = -1;
    read_tmp.strb = -1;    
    read_tmp.last = 0;

    if ((rptr == wptr_l) && ( wptr_done == true) ) {
      read_tmp.last = 1;
      wptr_l = 0;
      wptr_done = false;
      data_out.write(read_tmp);
      break;
    } else {
      read_tmp.last = 0;
      data_out.write(read_tmp);
    }   
  }
}

