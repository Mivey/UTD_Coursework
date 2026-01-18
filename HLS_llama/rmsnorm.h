#ifndef MARK_RMS
#define MARK_RMS
#include "forward.h"

template<int N_DIM>
void rmsnorm_kernel(s_mfdata_v_t &o, s_mfdata_v_t &x, s_mfdata_v_t &w){
  // #pragma HLS DATAFLOW
	constexpr int UF = 2;
  mfdata_v_t arr[N_DIM/MAX_FL_ELEM] = {0};
#pragma HLS ARRAY_PARTITION variable=arr type=complete
  mfdata_v_t ss = 0.0f;// = {0.0f}; // <----- added init value 0.0f 10/3 while working on MHA
  
  rms_mac_loop:
  for (int i = 0; i < (N_DIM / MAX_FL_ELEM); i++) {
    #pragma HLS PIPELINE
    mfdata_v_t tempval = x.read();
    ss += tempval * tempval;
		arr[i] = tempval;
  }

  my_float_t fss = 0.0f;
  // fdata_v_t tmp_ss = ss.read();
  adder_creation_loop:
  for (int i = 0; i < MAX_FL_ELEM; i++) {
    #pragma HLS UNROLL
    fss += ss[i];
  }
  fss /= (N_DIM);
  fss += 1e-5f;
  fss = 1.0f/hls::sqrtf(fss);

  // fdata_v_t tmp_o;
  data_out_loop:
  for (int i = 0 ; i < N_DIM/MAX_FL_ELEM; i++) {
    // #pragma HLS PIPELINE II=1
    mfdata_v_t tmp_o;
		mfdata_v_t temp_a = w.read();
		mfdata_v_t temp_b = arr[i];
		for (int j = 0; j < MAX_FL_ELEM; j+= UF) {
			#pragma HLS PIPELINE //II=3
			for (int k = 0 ; k < UF; k++) {
				tmp_o[j + k] = temp_a[k + j] * temp_b[k + j] * fss;
			}
		} 
    o.write(tmp_o);
  }
  // o.write(tmp_o);
}


#endif