#include "forward.h"
#include "matmul.h"
#include <cstdint>

void top(	
        mfdata_v_t *logits_out, mfdata_v_t *tokens, mfdata_v_t *key_cache, mfdata_v_t *value_cache,
        mfdata_v_t *rms_att_w, mfdata_v_t *rms_ffn_w, mfdata_v_t *rms_final_w,
        fdata_v_t *query_weights_sf, idata_v_t *query_weights_q,
        fdata_v_t *key_weights_sf, idata_v_t *key_weights_q,
        fdata_v_t *value_weights_sf, idata_v_t *value_weights_q,
        fdata_v_t *mha_output_weights_sf, idata_v_t *mha_output_weights_q,
        fdata_v_t *mlp_exp_weights1_sf, idata_v_t *mlp_exp_weights1_q,
        fdata_v_t *mlp_exp_weights3_sf, idata_v_t *mlp_exp_weights3_q,
        fdata_v_t *swiglu_comp_weights_sf, idata_v_t *swiglu_comp_weights_q,
        fdata_v_t *wcls_weights_sf, idata_v_t *wcls_weights_q,
        int POS, int LAYERS){
					
	// constexpr int MODEL_NUM_LAYERS
	constexpr int QUANT_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS / MAX_QUANT_ELEM;
	constexpr int SF_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS  / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int RMS_DEPTH = MODEL_ELEMENTS * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr	int LOGITS_QUANT_DEPTH = MODEL_ELEMENTS * MODEL_TOKENS / MAX_QUANT_ELEM;
	constexpr int LOGITS_SF_DEPTH =  MODEL_ELEMENTS * MODEL_TOKENS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int LOGITS_DEPTH = MODEL_TOKENS / MAX_FL_ELEM;

	#pragma HLS INTERFACE mode=m_axi port=value_cache							bundle=gmem8		depth=CACHE_DEPTH			offset=slave max_read_burst_length=128		max_write_burst_length=4
	#pragma HLS INTERFACE mode=m_axi port=key_cache								bundle=gmem9		depth=CACHE_DEPTH			offset=slave max_read_burst_length=128		max_write_burst_length=4
	#pragma HLS INTERFACE mode=m_axi port=tokens									bundle=gmem6	depth=TOK_DEPTH		offset=slave max_read_burst_length=128	
	#pragma HLS INTERFACE mode=m_axi port=rms_att_w								bundle=gmem7 	depth=RMS_DEPTH		offset=slave max_read_burst_length=128
	#pragma HLS INTERFACE mode=m_axi port=rms_final_w							bundle=gmem7 	depth=RMS_DEPTH	 	offset=slave max_read_burst_length=128
	#pragma HLS INTERFACE mode=m_axi port=query_weights_sf				bundle=gmem0 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
	#pragma HLS INTERFACE mode=m_axi port=query_weights_q					bundle=gmem1 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=256
	#pragma HLS INTERFACE mode=m_axi port=key_weights_sf					bundle=gmem2 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
	#pragma HLS INTERFACE mode=m_axi port=key_weights_q						bundle=gmem3 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=128
	#pragma HLS INTERFACE mode=m_axi port=value_weights_sf				bundle=gmem4 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
	#pragma HLS INTERFACE mode=m_axi port=value_weights_q					bundle=gmem5 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=128
	/* *************************************************************************	******************************************************/
	#pragma HLS INTERFACE mode=m_axi port=mha_output_weights_sf 	bundle=gmem0 	depth=SF_DEPTH 				offset=slave max_read_burst_length=16		
	#pragma HLS INTERFACE mode=m_axi port=mha_output_weights_q 		bundle=gmem1 	depth=QUANT_DEPTH 		offset=slave max_read_burst_length=256		
	/* *************************************************************************	*********/
	#pragma HLS INTERFACE mode=m_axi port=rms_ffn_w 							bundle=gmem7 	depth=RMS_DEPTH 	offset=slave max_read_burst_length=128	
	#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights1_sf 		bundle=gmem2 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=16	
	#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights1_q 			bundle=gmem3 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=128	
	#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights3_sf 		bundle=gmem4 	depth=HD_SF_DEPTH			offset=slave max_read_burst_length=16	
	#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights3_q 			bundle=gmem5 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=128	
	#pragma HLS INTERFACE mode=m_axi port=swiglu_comp_weights_sf 	bundle=gmem0 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=16	
	#pragma HLS INTERFACE mode=m_axi port=swiglu_comp_weights_q 	bundle=gmem1 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=256	
	#pragma HLS INTERFACE mode=m_axi port=wcls_weights_sf 				bundle=gmem0 	depth=LOGITS_SF_DEPTH 						offset=slave max_read_burst_length=16	
	#pragma HLS INTERFACE mode=m_axi port=wcls_weights_q 					bundle=gmem1 	depth=LOGITS_QUANT_DEPTH 						offset=slave max_read_burst_length=256
	#pragma HLS INTERFACE mode=m_axi port=logits_out 							bundle=gmem6 	depth=LOGITS_QUANT_DEPTH 						offset=slave max_write_burst_length=128		
	/* **********************************************************************************/
	#pragma HLS INTERFACE mode=s_axilite port=tokens				 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=rms_ffn_w 							bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mlp_exp_weights1_sf 		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mlp_exp_weights1_q 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mlp_exp_weights3_sf 		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mlp_exp_weights3_q 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=swiglu_comp_weights_sf 	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=swiglu_comp_weights_q 	bundle=control

	#pragma HLS INTERFACE mode=s_axilite port=LAYERS		 							bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=POS 										bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return 									bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_cache 						bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache								bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mha_output_weights_sf 	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mha_output_weights_q 		bundle=control
	/* **********************************************************************************/
	#pragma HLS INTERFACE mode=s_axilite port=rms_att_w								bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=query_weights_sf				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=query_weights_q					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_weights_sf					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_weights_q						bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_weights_sf				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_weights_q					bundle=control

	#pragma HLS INTERFACE mode=s_axilite port=rms_final_w					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=wcls_weights_sf				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=wcls_weights_q					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=logits_out					bundle=control

/* ************* Inter process streems ************************ */
	s_mfdata_v_t first_to_mha_tokens_stream("Tokens from first stream to MHA");
	s_mfdata_v_t mha_to_final_tokens_stream("Tokens from MHA to Final stream");
	s_mfdata_v_t query_stream("Query input Stream");
	// #pragma HLS BIND_STORAGE variable=first_to_mha_tokens_stream type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=mha_to_final_tokens_stream type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=query_stream type=fifo impl=lutram

	/* *********************** AXI top to subfunction streams ************ */

	s_mfdata_v_t s_tokens_in("Stream data for AXI Top tokens input"); 
	s_mfdata_v_t s_tokens_out("Stream data for AXI Top tokens output"); 
	s_mfdata_v_t s_tokens_out_loop("Stream data for AXI Top tokens loop output"); 
	s_mfdata_v_t s_tokens_out_exit("Stream data for AXI Top tokens exit output"); 
	s_mfdata_v_t s_key_cache_out("Stream data for AXI Top key_cache output");
	s_mfdata_v_t s_value_cache_out("Stream data for AXI Top value_cache output"); 
	s_mfdata_v_t s_key_cache_in("Stream data for AXI Top key_cache input");
	s_mfdata_v_t s_value_cache_in("Stream data for AXI Top value_cache input");
	s_mfdata_v_t s_rms_att_w("Stream data for AXI Top rms_att_w");
	s_mfdata_v_t s_rms_ffn_w("Stream data for AXI Top rms_ffn_w");
	s_fdata_v_t s_query_weights_sf("Stream data for AXI Top query_weights_sf");
	s_idata_v_t s_query_weights_q("Stream data for AXI Top query_weights_q");
	s_fdata_v_t s_key_weights_sf("Stream data for AXI Top key_weights_sf");
	s_idata_v_t s_key_weights_q("Stream data for AXI Top key_weights_q");
	s_fdata_v_t s_value_weights_sf("Stream data for AXI Top value_weights_sf");
	s_idata_v_t s_value_weights_q("Stream data for AXI Top value_weights_q");
	s_fdata_v_t s_mha_output_weights_sf("Stream data for AXI Top mha_output_weights_sf");
	s_idata_v_t s_mha_output_weights_q("Stream data for AXI Top mha_output_weights_q");
	s_fdata_v_t s_mlp_exp_weights1_sf("Stream data for AXI Top mlp_exp_weights1_sf");
	s_idata_v_t s_mlp_exp_weights1_q("Stream data for AXI Top mlp_exp_weights1_q");
	s_fdata_v_t s_mlp_exp_weights3_sf("Stream data for AXI Top mlp_exp_weights3_sf");
	s_idata_v_t s_mlp_exp_weights3_q("Stream data for AXI Top mlp_exp_weights3_q");
	s_fdata_v_t s_swiglu_comp_weights_sf("Stream data for AXI Top swiglu_comp_weights_sf");
	s_idata_v_t s_swiglu_comp_weights_q("Stream data for AXI Top swiglu_comp_weights_q");

	
	s_fdata_v_t s_first_tok_sf("Quantizer to MM First Kernel SF");
	s_idata_v_t s_first_tok_q("Quantizer to MM First Kernel quant");
	s_fdata_v_t s_final_tok_sf("Quantizer to MM Final Kernel SF");
	s_idata_v_t s_final_tok_q("Quantizer to MM Final Kernel quant");
	s_fdata_v_t s_logits_tok_sf("Quantizer to MM logits Kernel SF");
	s_idata_v_t s_logits_tok_q("Quantizer to MM logits Kernel quant");
	s_mfdata_v_t s_final_tokens_df("Tokens for the Final Kernel");

	#pragma HLS STREAM variable=first_to_mha_tokens_stream 		depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=mha_to_final_tokens_stream		depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=query_stream 									depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_final_tokens_df							depth=(MODEL_ELEMENTS / MAX_FL_ELEM)

	#pragma HLS STREAM variable=s_tokens_in depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	
	#pragma HLS STREAM variable=s_first_tok_sf depth=(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma HLS STREAM variable=s_final_tok_sf depth=(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM))
	#pragma HLS STREAM variable=s_first_tok_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	#pragma HLS STREAM variable=s_final_tok_q depth=(MODEL_ELEMENTS / MAX_QUANT_ELEM)
	
	#pragma HLS STREAM variable=s_key_cache_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_value_cache_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)

	tok_load_input(s_tokens_in, tokens, s_tokens_out_loop, 0);

	TOP_KERNEL_LOOP:	
	for (int CURR_LAYER = 0; CURR_LAYER < LAYERS; CURR_LAYER++) {
		#pragma hls LOOP_TRIPCOUNT max=12
		// #pragma HLS DATAFLOW
		
		// first_kernel_seq(first_to_mha_tokens_stream, s_first_tok_sf, s_first_tok_q, rms_att_w, s_tokens_in, CURR_LAYER);
		// first_kernel_dataflow(s_key_cache_out, s_value_cache_out, query_stream, 
		// 	s_first_tok_sf, s_first_tok_q, 
		// 	key_weights_sf, key_weights_q, 
		// 	value_weights_sf, value_weights_q, 
		// 	query_weights_sf, query_weights_q, 
		// 	CURR_LAYER, POS);
		first_kernel_seq(first_to_mha_tokens_stream, query_stream, s_key_cache_out, s_value_cache_out, key_weights_sf, key_weights_q, value_weights_sf, value_weights_q, query_weights_sf, query_weights_q, rms_att_w, s_tokens_in, CURR_LAYER, POS);
		
		mha_kernel_dataflow(mha_to_final_tokens_stream, 
			key_cache, value_cache, first_to_mha_tokens_stream, 
			s_key_cache_out, s_value_cache_out, 
			mha_output_weights_sf, mha_output_weights_q, 
			query_stream, POS, CURR_LAYER);
		
		// final_kernel_seq_a(s_final_tokens_df, s_final_tok_sf, s_final_tok_q, rms_ffn_w, mha_to_final_tokens_stream, CURR_LAYER);
		// final_third_dataflow(s_tokens_in, s_final_tokens_df, 
		// 	s_final_tok_sf, s_final_tok_q, 
		// 	mlp_exp_weights1_sf, mlp_exp_weights1_q, 
		// 	mlp_exp_weights3_sf, mlp_exp_weights3_q, 
		// 	swiglu_comp_weights_sf, swiglu_comp_weights_q, 
		// 	CURR_LAYER);
		final_third_seq_b(s_tokens_in, mha_to_final_tokens_stream, mlp_exp_weights1_sf, mlp_exp_weights1_q, mlp_exp_weights3_sf, mlp_exp_weights3_q, swiglu_comp_weights_sf, swiglu_comp_weights_q, rms_ffn_w, CURR_LAYER);
	}

	// logits_kernel(s_tokens_out, s_tokens_in, rms_final_w, wcls_weights_sf, wcls_weights_q);
	// logits_kernel_seq(s_logits_tok_sf, s_logits_tok_q, rms_final_w, s_tokens_in);
	// logits_kernel_dataflow(s_tokens_out, s_logits_tok_sf, s_logits_tok_q, wcls_weights_sf, wcls_weights_q);
	// store_output(tokens, s_tokens_in, MODEL_ELEMENTS, 0);// this is wrong. 
	logits_kernel(logits_out, rms_final_w, wcls_weights_sf, wcls_weights_q, s_tokens_in);
	return;
}