#include "forward.h"
#include "matmul.h"
#include "quantizer.h"
#include "rmsnorm.h"
void logits_kernel(
	mfdata_v_t *logits_out, mfdata_v_t *rms_final_w, 
	fdata_v_t *wcls_weights_sf, idata_v_t *wcls_weights_q, 
	s_mfdata_v_t &tokens_in){
	
	s_fdata_v_t tok_sf;
	s_idata_v_t tok_q;
	
	#pragma HLS STREAM variable=tok_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable=tok_sf depth=(MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR))

	rms_quant_router(tok_sf, tok_q, rms_final_w, tokens_in, 0);
	logits_matmult_kernel(logits_out, tok_sf, tok_q, wcls_weights_sf, wcls_weights_q, MODEL_ELEMENTS, MODEL_TOKENS, 0);
	}

// void logits_kernel_dataflow(s_mfdata_v_t &logits_out, 
// 	mfdata_v_t *rms_final_w, s_mfdata_v_t &tokens_in,
// 	fdata_v_t *wcls_weights_sf, idata_v_t *&wcls_weights_q){
		
// 	constexpr int LOG_TOK = MODEL_TOKENS * MODEL_ELEMENTS;
// 	constexpr int LOG_SF = LOG_TOK / MODEL_SCALING_FACTOR;
// 	// s_fdata_v_t s_wcls_weights_sf("wcls weights sf");
// 	// s_idata_v_t s_wcls_weights_q("wcls weights quant");
// 	s_fdata_v_t tok_sf;
// 	s_idata_v_t tok_q;
	
// 	#pragma HLS INLINE
	
// 	rms_quant_router(tok_sf, tok_q, rms_final_w, tokens_in, 0);	
// 	logits_matmult_kernel(mfdata_v_t *out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER)
	
// 	// mm_load_input(s_wcls_weights_sf, wcls_weights_sf, LOG_SF, 0); //MODEL_TOKENS / (SM_FL_ELEM * MODEL_SCALING_FACTOR), 0);
// 	// mm_tok_load_input(s_wcls_weights_q, wcls_weights_q, LOG_TOK, 0); //MODEL_ELEMENTS * MODEL_TOKENS / (MAX_QUANT_ELEM), 0);

// 	// matmult_kernel(logits_out, tok_sf, tok_q, s_wcls_weights_sf, s_wcls_weights_q, MODEL_ELEMENTS, MODEL_TOKENS);?
// }

