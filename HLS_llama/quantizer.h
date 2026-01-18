
#ifndef MARK_QUANTIZER
#define MARK_QUANTIZER
#include "forward.h"
#include <cstddef>
#include <hls_math.h>


template<size_t N_DIM>
void quantizer_kernel(s_fdata_v_t &tok_sf_out, s_idata_v_t &tok_out, s_mfdata_v_t &tokens){
	
	const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / MAX_FL_ELEM;
	const my_float_t Q_MAX = 127.0f;
	
	mfdata_v_t tok_arr[TOK_COUNT];
	#pragma HLS ARRAY_PARTITION variable=tok_arr dim=1 type=complete 
	fdata_v_t tmp_sf_out;
	
	quantizer_main_loop:
	for (size_t i = 0; i < SF_COUNT; i++) {
		my_float_t max_val = 0.0f;
		
		group_scaling_loop:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			// #pragma HLS PIPELINE II=1
			
			mfdata_v_t val = tokens.read();
			tok_arr[j] = val;

			quantizer_abs_val_loop:
			for (size_t k = 0; k < MAX_FL_ELEM; k++) {	
				#pragma HLS PIPELINE
				my_float_t a_val = hls::absf(val[k]);
				if (max_val < a_val) {max_val = a_val; }
			}
		}
		
		my_float_t dscale = max_val / Q_MAX; // will it matter if I set qmax = 1.0f/127 and them multiply?
		my_float_t scale = Q_MAX / max_val;
		const int QUANT_ARR_CNT = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;// * MAX_FL_ELEM;
		idata_v_t quant_tmp_arr[QUANT_ARR_CNT];
	#pragma HLS ARRAY_PARTITION variable=quant_tmp_arr dim=1 type=complete 
		
		create_quant_val_loop:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			mfdata_v_t proc_tok = tok_arr[j];// * scale;
			int n = (MAX_FL_ELEM * j) % MAX_QUANT_ELEM;
			int m = (MAX_FL_ELEM * j) / MAX_QUANT_ELEM;
				
			for (size_t k = 0; k < MAX_FL_ELEM; k++) {
			#pragma HLS PIPELINE
				quant_tmp_arr[m][n + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
			}
		}
		
		for (int i = 0; i < QUANT_ARR_CNT; i++) {
			#pragma HLS UNROLL
			tok_out.write(quant_tmp_arr[i]);
		}
		tmp_sf_out[i % SM_FL_ELEM] = dscale;
		
		if ((i % SM_FL_ELEM) == (SM_FL_ELEM - 1)) {
			tok_sf_out.write(tmp_sf_out);
		}
	}
}

#endif
