
#include "forward.h"
#include "quantizer.h"
#include "rmsnorm.h"
#include <cstdint>

constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM));

typedef hls::vector<int16_t, MAX_QUANT_ELEM> two_byte_data_v_t;

void send_quant_wtok(hls::stream<my_float_t> &wtok, s_idata_v_t &w, s_idata_v_t &tok, const int N_DIM, const int M_DIM){
	const int num_sf_per_col = N_DIM / MODEL_SCALING_FACTOR;
	const int vCount = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;
	const int TOK_ARR_SIZE =  num_sf_per_col * vCount;//N_DIM / MAX_QUANT_ELEM;
	idata_v_t arr[TOK_QUANT_MAX];
	#pragma HLS ARRAY_PARTITION variable=arr complete dim=1

	get_tok_loop:
	for(size_t i = 0; i < (TOK_ARR_SIZE); i++){
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_TRIPCOUNT max = TOK_QUANT_MAX min=MODEL_ELEMENTS/(MAX_QUANT_ELEM)  
		arr[i] = tok.read();
	}	

	send_wtok_loop:
	for (size_t i = 0; i < M_DIM; i++){
		#pragma HLS LOOP_TRIPCOUNT max = MODEL_HIDDEN_DIM min = MODEL_ELEMENTS
		for (size_t j = 0; j < (num_sf_per_col); j++){
			#pragma HLS LOOP_TRIPCOUNT max = TOK_QUANT_MAX min=MODEL_ELEMENTS/(MAX_QUANT_ELEM) 
			// #pragma HLS PIPELINE II=4
			int32_t sum_temp = 0;
			for (int k = 0; k < vCount; k++) {
				#pragma HLS PIPELINE II=1
				int m = j * vCount + k;
				idata_v_t wtemp = w.read();
				idata_v_t tok_val = arr[m];

				for (int l = 0; l < MAX_QUANT_ELEM; l++) {
					sum_temp += tok_val[l] * wtemp[l];
				}
			}
			wtok.write((my_float_t) sum_temp);			
		}
	}
}

void send_sf_wtok(s_fdata_v_t &wtok_sf, s_fdata_v_t &w_sf, s_fdata_v_t &tok_sf, const int N_DIM, const int M_DIM){
	const int SF_ARR_SIZE = (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM));
	const int sfCount = N_DIM / (SM_FL_ELEM * MODEL_SCALING_FACTOR);
	fdata_v_t arr[TOK_SF_MAX];
	#pragma HLS ARRAY_PARTITION variable=arr complete dim=1

	
	get_tok_sf_loop:
	for (size_t i = 0; i < sfCount; i++){ // vCount here is 1/4 vCount in send_wtok!!
		#pragma HLS PIPELINE II=1
  	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM)  

		arr[i] = tok_sf.read();	
	}

	send_wtok_sf_total_loop:
	for (size_t i = 0; i < M_DIM; i++){
  	#pragma HLS LOOP_TRIPCOUNT max = MODEL_HIDDEN_DIM min = MODEL_ELEMENTS
		// #pragma HLS PIPELINE
				
  	send_wtok_sf_elem_loop:
		for (size_t j = 0; j < sfCount; j++){
			#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM)  
			#pragma HLS PIPELINE II=1
			// #pragma HLS UNROLL
			fdata_v_t wtmp = w_sf.read();
			fdata_v_t temp = arr[j] * wtmp;
			wtok_sf.write(temp);
		}	
	}
}

void mat_mult_main(s_mfdata_v_t &out, s_fdata_v_t &wtok_sf, hls::stream<my_float_t> &wtok, const int N_DIM, const int M_DIM){

	const int sfCount = N_DIM / (SM_FL_ELEM * MODEL_SCALING_FACTOR);
			mfdata_v_t temp_out;//, temp_mul;
  main_calc_loop:
	for (size_t i = 0; i < M_DIM; i++){
    #pragma HLS LOOP_TRIPCOUNT max = MODEL_HIDDEN_DIM min = MODEL_ELEMENTS

		my_float_t tmp_sum = 0.0f;
    elem_calc_loop:
		for (size_t j = 0; j < sfCount; j++){
      #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM)  
			#pragma HLS PIPELINE
			
			fdata_v_t wtok_tmp;
      fdata_v_t temp_sf = wtok_sf.read();
			
			for (size_t k = 0; k < SM_FL_ELEM; k++){
				// #pragma HLS PIPELINE II=1
				wtok_tmp[k] = wtok.read() * temp_sf[k];
			}
            
			tmp_sum += wtok_tmp.reduce_add();
			if (j == (sfCount - 1)) {
				temp_out[(i % MAX_FL_ELEM)] = tmp_sum;
				tmp_sum = 0.0f;
			}
		}
		if ((i % MAX_FL_ELEM) == (MAX_FL_ELEM - 1)){
			out.write(temp_out);
			// std::fill(temp_out.begin(), temp_out.end(), 0.0f);
		}
	}
}

void matmult_kernel(s_mfdata_v_t &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf, s_idata_v_t &w, int N_DIM, int M_DIM){

	hls::stream<my_float_t> wtok("wtok causing problems");
	s_fdata_v_t wtok_sf;
	#pragma hls STREAM variable = wtok_sf //depth = 3
	#pragma HLS STREAM variable = wtok //depth = 12
  #pragma HLS DATAFLOW
    
	send_quant_wtok(wtok, w, tok, N_DIM, M_DIM);
	send_sf_wtok(wtok_sf, w_sf, tok_sf, N_DIM, M_DIM);
  mat_mult_main(out, wtok_sf, wtok, N_DIM, M_DIM);
}

void two_matmult_kernel(s_mfdata_v_t &out1, s_mfdata_v_t &out2, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf1, idata_v_t *w1, fdata_v_t *w_sf2, idata_v_t *w2, const int N_DIM, const int M_DIM, const int CURR_LAYER){
	
	
	s_fdata_v_t tok_sf1, tok_sf2;
	s_idata_v_t tok1, tok2;
	s_fdata_v_t s_wsf1, s_wsf2;
	s_idata_v_t s_w1, s_w2;
	const int num = N_DIM * M_DIM;
	const int num_sf = num / MODEL_SCALING_FACTOR;
	
	#pragma hls STREAM variable = tok_sf1 depth = 4 //(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma hls STREAM variable = tok_sf2 depth = 4 //(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma HLS STREAM variable = tok1 depth = 4 //(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable = tok2 depth = 4 //(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	// #pragma HLS BIND_STORAGE variable=tok_sf1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok_sf2 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok2 type=fifo impl=lutram
	
  #pragma HLS DATAFLOW
	mm_load_input(s_wsf1, w_sf1, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w1, w1, num, CURR_LAYER);
	
	mm_load_input(s_wsf2, w_sf2, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w2, w2, num, CURR_LAYER);
	
	split_tee<fdata_v_t>(tok_sf1, tok_sf2, tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	split_tee(tok1, tok2, tok, (N_DIM / MAX_QUANT_ELEM));

	matmult_kernel(out1, tok_sf1, tok1, s_wsf1, s_w1, N_DIM, M_DIM);
	matmult_kernel(out2, tok_sf2, tok2, s_wsf2, s_w2, N_DIM, M_DIM);
}
void final_two_matmult_kernel(s_fdata_v_t &hb_sf, s_idata_v_t &hb_tok, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf1, idata_v_t *w1, fdata_v_t *w_sf2, idata_v_t *w2, const int N_DIM, const int M_DIM, const int CURR_LAYER){
	
	
	s_fdata_v_t tok_sf1, tok_sf2;
	s_idata_v_t tok1, tok2;
	s_fdata_v_t s_wsf1, s_wsf2;
	s_idata_v_t s_w1, s_w2;
	s_mfdata_v_t out1, out2, hb_out;
	const int num = N_DIM * M_DIM;
	const int num_sf = num / MODEL_SCALING_FACTOR;
	
	#pragma hls STREAM variable = tok_sf1 depth = 4 //(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma hls STREAM variable = tok_sf2 depth = 4 //(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma HLS STREAM variable = tok1 depth = 4 //(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable = tok2 depth = 4 //(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	// #pragma HLS BIND_STORAGE variable=tok_sf1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok_sf2 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok2 type=fifo impl=lutram
	
  #pragma HLS DATAFLOW
	mm_load_input(s_wsf1, w_sf1, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w1, w1, num, CURR_LAYER);
	
	mm_load_input(s_wsf2, w_sf2, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w2, w2, num, CURR_LAYER);
	
	split_tee<fdata_v_t>(tok_sf1, tok_sf2, tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	split_tee(tok1, tok2, tok, (N_DIM / MAX_QUANT_ELEM));

	matmult_kernel(out1, tok_sf1, tok1, s_wsf1, s_w1, N_DIM, M_DIM);
	matmult_kernel(out2, tok_sf2, tok2, s_wsf2, s_w2, N_DIM, M_DIM);

	swiglu_kernel<MODEL_HIDDEN_DIM>(hb_out, out1, out2);
	quantizer_kernel<MODEL_HIDDEN_DIM>(hb_sf, hb_tok, hb_out);
}

void one_matmult_kernel(s_mfdata_v_t &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER){
	
	s_fdata_v_t s_sf;
	s_idata_v_t s_w;
	#pragma HLS DATAFLOW
	
	const int num = N_DIM * M_DIM;
	const int num_sf = num / MODEL_SCALING_FACTOR;
	mm_load_input(s_sf, w_sf, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w, w, num, CURR_LAYER);

	matmult_kernel(out, tok_sf, tok, s_sf, s_w, N_DIM, M_DIM);
}

void logits_matmult_kernel(mfdata_v_t *out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER){
	
	s_fdata_v_t s_sf;
	s_idata_v_t s_w;
	s_mfdata_v_t logits_out;
	#pragma HLS DATAFLOW
	
	const int num = N_DIM * M_DIM;
	const int num_sf = num / MODEL_SCALING_FACTOR;
	mm_load_input(s_sf, w_sf, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w, w, num, CURR_LAYER);

	matmult_kernel(logits_out, tok_sf, tok, s_sf, s_w, N_DIM, M_DIM);
	store_output(out, logits_out, MODEL_TOKENS, CURR_LAYER);
}
void two_matmult_sg_kernel(/*s_mfdata_v_t &hb_out,*/ s_fdata_v_t &hb_sf, s_idata_v_t &hb_tok, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf1, s_idata_v_t &w1, s_fdata_v_t &w_sf2, s_idata_v_t &w2){

	s_fdata_v_t tok_sf1, tok_sf2;
	s_idata_v_t tok1, tok2;
	s_mfdata_v_t out1, out2;
	s_mfdata_v_t hb_out;
	
	#pragma hls STREAM variable = tok_sf1 depth = 4//(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma hls STREAM variable = tok_sf2 depth = 4//(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma HLS STREAM variable = tok1 depth = 4//(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable = tok2 depth = 4//(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable = hb_out depth = 4
	// #pragma HLS BIND_STORAGE variable=tok_sf1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok_sf2 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok1 type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=tok2 type=fifo impl=lutram
	
  #pragma HLS INLINE
	
	split_tee<fdata_v_t>(tok_sf1, tok_sf2, tok_sf, (MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	split_tee(tok1, tok2, tok, (MODEL_ELEMENTS / MAX_QUANT_ELEM));

	matmult_kernel(out1, tok_sf1, tok1, w_sf1, w1, MODEL_ELEMENTS, MODEL_HIDDEN_DIM);
	matmult_kernel(out2, tok_sf2, tok2, w_sf2, w2, MODEL_ELEMENTS, MODEL_HIDDEN_DIM);
	swiglu_kernel<MODEL_HIDDEN_DIM>(hb_out, out1, out2);
	quantizer_kernel<MODEL_HIDDEN_DIM>(hb_sf, hb_tok, hb_out);
}

template<int HEAD>
void init_freq_arr(float arr[HEAD]){
  for (int i = 0; i < HEAD; i++) {
  arr[i] = 1.0f / hls::powf(10000.0f, ((i) / (float) MODEL_HEAD_SIZE));
  }
}

void rope_kernel (s_mfdata_v_t &out, s_mfdata_v_t &in, const int N_DIM, const int POS){
	// #pragma HLS DATAFLOW
	float arr[MODEL_HEAD_SIZE];
	init_freq_arr<MODEL_HEAD_SIZE>(arr);
	ROPE_MAIN:
	for (int i = 0; i < (N_DIM / MAX_FL_ELEM); i++) {
	
	int k = i * MAX_FL_ELEM;
		mfdata_v_t tmp = in.read();
		mfdata_v_t tmp_o;
		head_dim_unroll_loop:
		for (int j = 0 ; j < (MAX_FL_ELEM / 2); j++) {
			#pragma HLS PIPELINE
			// #pragma HLS UNROLL factor=8
			int head_dim = (k + j * 2) % MODEL_HEAD_SIZE;
			float freq =  arr[head_dim]; /*1.0f / hls::powf(10000.0f, (float)head_dim/HEAD_SIZE);*/ 
			float val = POS * freq;
			float fcr;
			float fci;
			hls::sincosf(val, &fci, &fcr);
			float v0 = (float) tmp[j * 2 + 0];
			float v1 = (float) tmp[j * 2 + 1];
			tmp_o[j * 2 + 0] = v0 * fcr - v1 * fci;
			tmp_o[j * 2 + 1] = v0 * fci + v1 * fcr;
		}
		out.write(tmp_o);
	}
}

void rms_quant_router( s_fdata_v_t &quant_sf, s_idata_v_t &quant_tok, mfdata_v_t *rms_w, s_mfdata_v_t &in_tokens, const int CURR_LAYER){

	constexpr int vCount = MODEL_ELEMENTS / MAX_FL_ELEM;
	// #pragma HLS STREAM variable=out_tokens	depth=vCount
	// #pragma HLS STREAM variable=quant_sf		depth=vCount * (MAX_FL_ELEM / SM_FL_ELEM) / MODEL_SCALING_FACTOR
	// #pragma HLS STREAM variable=quant_tok		depth=MODEL_ELEMENTS / MAX_QUANT_ELEM
	// #pragma HLS STREAM variable=rms_w				depth=vCount
	// #pragma HLS STREAM variable=in_tokens		depth=vCount

	// s_mfdata_v_t internal_tokens("Internal rmsnorm.cpp tokens");
	s_mfdata_v_t s_rms_w("Internal RMSnorm w stream");
	s_mfdata_v_t rms_to_quant("Internal rmsnorm kernel to quantzier in rmsnorm.cpp");
	#pragma HLS DATAFLOW
	rms_load_input(s_rms_w, rms_w, CURR_LAYER);
	// split_tee(out_tokens, internal_tokens, in_tokens, vCount);

	rmsnorm_kernel<MODEL_ELEMENTS>(rms_to_quant, in_tokens, s_rms_w);

	quantizer_kernel<MODEL_ELEMENTS>(quant_sf, quant_tok, rms_to_quant);
	
}