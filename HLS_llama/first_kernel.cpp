#include "forward.h"
#include "matmul.h"
#include "rmsnorm.h"
#include "quantizer.h"

void first_kernel_dataflow(
	s_mfdata_v_t &key_cache_out, s_mfdata_v_t &value_cache_out,
	s_mfdata_v_t &query_out, 
	s_fdata_v_t &tok_sf, s_idata_v_t &tok_q,
	fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
	fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
	fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
	const int CURR_LAYER, const int POS){
	

	s_fdata_v_t s_key_weights_sf; 
	s_idata_v_t s_key_weights_q;
	s_fdata_v_t s_value_weights_sf; 
	s_idata_v_t s_value_weights_q;
	s_fdata_v_t s_query_weights_sf; 
	s_idata_v_t s_query_weights_q; 

	s_fdata_v_t spl_wk_sf, spl_wv_sf, spl_wq_sf;
	s_idata_v_t spl_wk_q, spl_wv_q, spl_wq_q;
	s_mfdata_v_t r_key_cache, r_query;


  #pragma HLS STABLE variable=POS
  #pragma HLS STABLE variable=CURR_LAYER
	#pragma HLS DATAFLOW

	split_tee(spl_wk_sf, spl_wv_sf, spl_wq_sf, tok_sf, (MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	split_tee(spl_wk_q, spl_wv_q, spl_wq_q, tok_q, (MODEL_ELEMENTS / MAX_QUANT_ELEM));

	mm_load_input(s_key_weights_sf, key_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_key_weights_q, key_weights_q, SQUARE_TOK, CURR_LAYER);
	matmult_kernel(r_key_cache, spl_wk_sf, spl_wk_q, s_key_weights_sf, s_key_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);
	
	mm_load_input(s_value_weights_sf, value_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_value_weights_q, value_weights_q, SQUARE_TOK, CURR_LAYER);
	matmult_kernel(value_cache_out, spl_wv_sf, spl_wv_q, s_value_weights_sf, s_value_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);
	
	mm_load_input(s_query_weights_sf, query_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_query_weights_q, query_weights_q, SQUARE_TOK, CURR_LAYER);
	matmult_kernel(r_query, spl_wq_sf, spl_wq_q, s_query_weights_sf, s_query_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);

	rope_kernel(key_cache_out, r_key_cache, MODEL_ELEMENTS, POS);
	rope_kernel(query_out, r_query, MODEL_ELEMENTS, POS);
	
	}

void first_kernel_seq(
	s_mfdata_v_t &tokens_out, s_mfdata_v_t &query_out, 
	s_mfdata_v_t &key_cache_out, s_mfdata_v_t &value_cache_out,
	fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
	fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
	fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
	mfdata_v_t *rms_att_w, s_mfdata_v_t &tokens_in, 
	const int CURR_LAYER, const int POS){
	
	// s_mfdata_v_t s_rms_att_w("rms attention weights");
	s_mfdata_v_t s_tokens_rms_att("Tokens Input to RMSNorm");
	s_mfdata_v_t s_rms_att_to_quant("RMS Norm to Quantizer kernel");
	s_mfdata_v_t r_key_cache, r_query;

	s_fdata_v_t tok_sf, spl_wk_sf, spl_wq_sf;
	s_idata_v_t tok_q, spl_wk_q, spl_wq_q;

	#pragma HLS INLINE
  #pragma HLS STABLE variable=CURR_LAYER
	// #pragma HLS STREAM variable=s_rms_att_w depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_tokens_rms_att depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=tokens_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_rms_att_to_quant depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=tok_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable=spl_wk_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable=spl_wq_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable=tok_sf depth=(MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR))
	#pragma HLS STREAM variable=spl_wk_sf depth=(MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR))
	#pragma HLS STREAM variable=spl_wq_sf depth=(MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR))
	#pragma HLS STREAM variable=r_key_cache depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=r_query depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=key_cache_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=value_cache_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=query_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)

	// splitter<mfdata_v_t, MODEL_ELEMENTS / MAX_FL_ELEM>(s_tokens_rms_att, tokens_out, tokens_in);?
	split_tee(s_tokens_rms_att, tokens_out, tokens_in, (MODEL_ELEMENTS/MAX_FL_ELEM));
	
	rms_quant_router(tok_sf, tok_q, rms_att_w, s_tokens_rms_att, CURR_LAYER);
	split_tee(spl_wk_sf, spl_wq_sf, tok_sf, (MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	split_tee(spl_wk_q, spl_wq_q, tok_q, (MODEL_ELEMENTS / MAX_QUANT_ELEM));
	// two_matmult_kernel(r_key_cache, value_cache_out, spl_wk_sf, spl_wk_q, s_key_weights_sf, s_key_weights_q, s_value_weights_sf, s_value_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);
	two_matmult_kernel(r_key_cache, value_cache_out, spl_wk_sf, spl_wk_q, key_weights_sf, key_weights_q, value_weights_sf, value_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS, CURR_LAYER);
	
	// matmult_kernel(r_query, spl_wq_sf, spl_wq_q, s_query_weights_sf, s_query_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);
	one_matmult_kernel(r_query, spl_wq_sf, spl_wq_q, query_weights_sf, query_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS, CURR_LAYER);
	rope_kernel(key_cache_out, r_key_cache, MODEL_ELEMENTS, POS);
	rope_kernel(query_out, r_query, MODEL_ELEMENTS, POS);
}

void first_kernel_df(
	s_fdata_v_t &s_key_weights_sf, s_idata_v_t &s_key_weights_q,
	s_fdata_v_t &s_value_weights_sf, s_idata_v_t &s_value_weights_q,
	s_fdata_v_t &s_query_weights_sf, s_idata_v_t &s_query_weights_q,
	fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
	fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
	fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
	const int CURR_LAYER){

  // #pragma HLS STABLE variable=POS
  #pragma HLS STABLE variable=CURR_LAYER
	#pragma HLS DATAFLOW

	mm_load_input(s_key_weights_sf, key_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_key_weights_q, key_weights_q, SQUARE_TOK, CURR_LAYER);
	
	mm_load_input(s_value_weights_sf, value_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_value_weights_q, value_weights_q, SQUARE_TOK, CURR_LAYER);
	
	mm_load_input(s_query_weights_sf, query_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_query_weights_q, query_weights_q, SQUARE_TOK, CURR_LAYER);

	}