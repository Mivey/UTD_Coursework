
#ifndef MARK_FORWARD
#define MARK_FORWARD

// #include "fast_common.h"
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
// #include <ap_float.h>

#define DATAWIDTH 32
#define MODEL_ELEMENTS 768
#define MODEL_HIDDEN_DIM 2048
#define QUANT 8 // bits in the word... either 4 or 8
#define MODEL_NUM_HEADS 12
#define MODEL_NUM_LAYERS 12
#define MODEL_TOKENS 32000
#define MODEL_SEQUENCE_LEN 1024
#define MODEL_SCALING_FACTOR 64
#define bytes_in(n) sizeof(n)
#define runs(n) SCALING_FACTOR/sizeof(n)
constexpr float Q_FACTOR = ((QUANT%4)==0) ? \
                 static_cast<float>((1<<(QUANT - 1)) - 1) : 127;

/* ************************************* */
// typedef ap_float<32, 8> my_float_t;
typedef float my_float_t;
typedef int8_t my_quant_data_t;
/* ************************************* */

constexpr size_t MAX_DW = 512;
constexpr size_t SM_DW = 128;
constexpr size_t MAX_FL_ELEM = (MAX_DW / (sizeof(my_float_t) * 8));
constexpr size_t MAX_QUANT_ELEM = ((MAX_DW / 2) / (sizeof(my_quant_data_t) * 8));
constexpr size_t SM_FL_ELEM = (SM_DW / (sizeof(my_float_t) * 8));
constexpr size_t SM_QUANT_ELEM = (SM_DW / (sizeof(my_quant_data_t) * 8));

constexpr int MODEL_HEAD_SIZE = MODEL_ELEMENTS / MODEL_NUM_HEADS;
// #define MAX_W 512
constexpr int MAX_W_Q = MAX_DW/(sizeof(my_quant_data_t) * 8);
constexpr int MAX_W_F = MAX_DW/(sizeof(my_float_t) * 8);
constexpr int MAX_SF_W_F = MAX_DW/(sizeof(my_float_t) * 4 * 8);
constexpr int TOK_CHUNKSIZE = 256;
constexpr int MM_CHUNKSIZE = 256;
constexpr int MHA_CHUNKSIZE = 64;

const int SQUARE_TOK = MODEL_ELEMENTS * MODEL_ELEMENTS;
const int SQUARE_SF = SQUARE_TOK / MODEL_SCALING_FACTOR;
const int RECT_TOK = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
const int RECT_SF = RECT_TOK / MODEL_SCALING_FACTOR;

/* ==================================================================================== */



typedef hls::vector<my_quant_data_t, MAX_QUANT_ELEM> idata_v_t;
typedef hls::vector<my_float_t, SM_FL_ELEM>	fdata_v_t;
typedef hls::vector<my_float_t, MAX_FL_ELEM>	mfdata_v_t;

typedef hls::stream<idata_v_t> s_idata_v_t;
typedef hls::stream<fdata_v_t> s_fdata_v_t; 
typedef hls::stream<mfdata_v_t> s_mfdata_v_t;

template<typename T>
void split_tee(hls::stream<T> &out0, hls::stream<T> &out1, hls::stream<T> &in, const int vCount){
	
  for (int i = 0; i < vCount; i++) {
    #pragma HLS PIPELINE II=1
    T data = in.read();
    out0.write(data);
    out1.write(data);
	}
}
template<typename T>
void split_tee(hls::stream<T> &out0, hls::stream<T> &out1, hls::stream<T> &out2, hls::stream<T> &in, const int vCount){
	
  for (int i = 0; i < vCount; i++) {
    #pragma HLS PIPELINE II=1
    T data = in.read();
    out0.write(data);
    out1.write(data);
    out2.write(data);
	}
}

/* =================================== RESIDUAL CONNECTION ===================================== */

template<int N_DIM>
void resid_conn(s_mfdata_v_t &tokens_out, s_mfdata_v_t &tokens_in, s_mfdata_v_t &xb){
	for (int i = 0; i < N_DIM / MAX_FL_ELEM; i++) {
		#pragma HLS PIPELINE II=1
		mfdata_v_t tmp, tmpa, tmpb;
		tmpa =tokens_in.read();
		tmpb = xb.read();
		tmp = tmpa + tmpb;// tokens_in.read() + xb.read();
		tokens_out.write(tmp);

	}
}

/* *************************** SWIGLU FUNCTION *************************************/
template<int HID_DIM>
void swiglu_kernel(s_mfdata_v_t &hb_out, s_mfdata_v_t &hb_in, s_mfdata_v_t &hb2_in){
	for (int i = 0 ; i < HID_DIM / MAX_FL_ELEM; i++) {
	#pragma HLS pipeline II=4
		mfdata_v_t val =hb_in.read();
		mfdata_v_t eval;
		for (int j = 0; j < MAX_FL_ELEM; j++) {
			#pragma HLS UNROLL
			eval[j] = val[j] / ( 1.0f + hls::expf(-1 * (float)val[j]));
		}
		hb_out.write(eval * hb2_in.read());
	}
}

// void load_input(s_fdata_v_t &out, fdata_v_t *in, const int vSize);
void tok_load_input(s_mfdata_v_t &out, mfdata_v_t *in, s_mfdata_v_t &in_tokens, const int CURR_LAYER);
void rms_load_input(s_mfdata_v_t &out, mfdata_v_t *in, const int CURR_LAYER);
void mm_load_input(s_fdata_v_t &out, fdata_v_t *in, const int vCount, const int CURR_LAYER);
void mm_tok_load_input(s_idata_v_t &out, idata_v_t *in, const int vCount, const int CURR_LAYER);
void store_output(mfdata_v_t *out, s_mfdata_v_t &in , const int vSize, const int CURR_LAYER);
void mha_WAR_store_load(mfdata_v_t *cache, s_mfdata_v_t &output, s_mfdata_v_t &input, const int CURR_LAYER, const int POS);

void top(	
	//mfdata_v_t *output_tokens, 					mfdata_v_t *key_cache_out, 				mfdata_v_t *value_cache_out, // outputs
	mfdata_v_t *logits_out, 
  mfdata_v_t *tokens, 								mfdata_v_t *key_cache, 						mfdata_v_t *value_cache,
  mfdata_v_t *rms_att_w, 							mfdata_v_t *rms_ffn_w, 						mfdata_v_t *rms_final_w,
  fdata_v_t *query_weights_sf, 				idata_v_t *query_weights_q,
  fdata_v_t *key_weights_sf, 					idata_v_t *key_weights_q,
  fdata_v_t *value_weights_sf, 				idata_v_t *value_weights_q,
  fdata_v_t *mha_output_weights_sf, 	idata_v_t *mha_output_weights_q,
  fdata_v_t *mlp_exp_weights1_sf, 		idata_v_t *mlp_exp_weights1_q,
  fdata_v_t *mlp_exp_weights3_sf, 		idata_v_t *mlp_exp_weights3_q,
  fdata_v_t *swiglu_comp_weights_sf, 	idata_v_t *swiglu_comp_weights_q,
	fdata_v_t *wcls_weights_sf, 				idata_v_t *wcls_weights_q,
  int POS, 														int CURR_LAYER
	);

// void first_kernel_dataflow(
// 	s_mfdata_v_t &key_cache_out, s_mfdata_v_t &value_cache_out,
// 	s_mfdata_v_t &query_out, 
// 	s_fdata_v_t &tok_sf, s_idata_v_t &tok_q,
// 	fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
// 	fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
// 	fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
// 	const int CURR_LAYER, const int POS);
	
void first_kernel_seq(
	s_mfdata_v_t &tokens_out, s_mfdata_v_t &query_out, 
	s_mfdata_v_t &key_cache_out, s_mfdata_v_t &value_cache_out,
	fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
	fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
	fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
	mfdata_v_t *rms_att_w, s_mfdata_v_t &tokens_in, 
	const int CURR_LAYER, const int POS);
	
	// void first_kernel_df(
	// s_fdata_v_t &s_key_weights_sf, s_idata_v_t &s_key_weights_q,
	// s_fdata_v_t &s_value_weights_sf, s_idata_v_t &s_value_weights_q,
	// s_fdata_v_t &s_query_weights_sf, s_idata_v_t &s_query_weights_q,
	// fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
	// fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
	// fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
	// const int CURR_LAYER);

void mha_kernel_dataflow(s_mfdata_v_t &tokens_out, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                s_mfdata_v_t &tokens_in,
								s_mfdata_v_t &key_cache_in, s_mfdata_v_t &value_cache_in,
                fdata_v_t *mha_output_weights_sf, 
                idata_v_t *mha_output_weights_q, 
                s_mfdata_v_t &query, 
                const int POS, const int CURR_LAYER);

void final_kernel_seq_a(
	s_mfdata_v_t &tokens_out,
	s_fdata_v_t &tok_sf, s_idata_v_t &tok_q,
	mfdata_v_t *rms_ffn_w, s_mfdata_v_t &tokens_in, 
	const int CURR_LAYER);

void final_third_dataflow(
	s_mfdata_v_t &output_tokens,
	s_mfdata_v_t &input_tokens,
	s_fdata_v_t &tok_sf, s_idata_v_t &tok_q,
	fdata_v_t *mlp_exp_weights1_sf, idata_v_t *mlp_exp_weights1_q,
	fdata_v_t *mlp_exp_weights3_sf, idata_v_t *mlp_exp_weights3_q,
	fdata_v_t *swiglu_comp_weights_sf, idata_v_t *swiglu_comp_weights_q,
	const int CURR_LAYER);

	void final_third_seq_b(
	s_mfdata_v_t &output_tokens,
	s_mfdata_v_t &input_tokens,
	fdata_v_t *mlp_exp_weights1_sf, idata_v_t *mlp_exp_weights1_q,
	fdata_v_t *mlp_exp_weights3_sf, idata_v_t *mlp_exp_weights3_q,
	fdata_v_t *swiglu_comp_weights_sf, idata_v_t *swiglu_comp_weights_q,
	mfdata_v_t *rms_ffn_w,	const int CURR_LAYER);
void logits_kernel(
	mfdata_v_t *logits_out, mfdata_v_t *rms_final_w, 
	fdata_v_t *wcls_weights_sf, idata_v_t *wcls_weights_q, 
	s_mfdata_v_t &tokens_in);
	
template<typename T>
void dummy_read(hls::stream<T> &in, const int vCount){
	for (int i = 0; i < vCount; i++) {
		T boop = in.read();
	}
}
#endif