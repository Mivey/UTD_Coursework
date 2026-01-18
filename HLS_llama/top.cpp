#include "forward.h"
#include "matmul.h"

void top(	//mfdata_v_t *output_tokens, mfdata_v_t *key_cache_out, mfdata_v_t *value_cache_out, // outputs
        mfdata_v_t *tokens, mfdata_v_t *key_cache, mfdata_v_t *value_cache,
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
/* delete for real thing*/
// RUTL CO SIM WILL FAIL UNTIL I CHANGE ALL VALUES OF DEPTH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
// #pragma HLS INTERFACE mode=m_axi port=value_cache_out		 					bundle=gmem9b			depth=48		max_write_burst_length=8
// #pragma HLS INTERFACE mode=m_axi port=key_cache_out								bundle=gmem9a			depth=48		max_write_burst_length=8
// #pragma HLS INTERFACE mode=m_axi port=output_tokens								bundle=gmem6a		depth=48		max_write_burst_length=64
constexpr int QUANT_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS / MAX_QUANT_ELEM;
constexpr int SF_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS / MAX_QUANT_ELEM;
constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_NUM_LAYERS * 12 * MODEL_SEQUENCE_LEN / MAX_FL_ELEM;
constexpr int TOK_RMS_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;

#pragma HLS INTERFACE mode=m_axi port=value_cache							bundle=gmem9b		depth=CACHE_DEPTH			offset=slave max_read_burst_length=128		max_write_burst_length=4
#pragma HLS INTERFACE mode=m_axi port=key_cache								bundle=gmem9a		depth=CACHE_DEPTH			offset=slave max_read_burst_length=128		max_write_burst_length=4
#pragma HLS INTERFACE mode=m_axi port=tokens									bundle=gmem6a 	depth=TOK_RMS_DEPTH		offset=slave max_read_burst_length=128		max_write_burst_length=128
#pragma HLS INTERFACE mode=m_axi port=rms_att_w								bundle=gmem7ab 	depth=TOK_RMS_DEPTH		offset=slave max_read_burst_length=128
#pragma HLS INTERFACE mode=m_axi port=rms_final_w							bundle=gmem7a 	depth=TOK_RMS_DEPTH	 	offset=slave max_read_burst_length=128
#pragma HLS INTERFACE mode=m_axi port=query_weights_sf				bundle=gmem0 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
#pragma HLS INTERFACE mode=m_axi port=query_weights_q					bundle=gmem1 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=256
#pragma HLS INTERFACE mode=m_axi port=key_weights_sf					bundle=gmem2 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
#pragma HLS INTERFACE mode=m_axi port=key_weights_q						bundle=gmem3 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=128
#pragma HLS INTERFACE mode=m_axi port=value_weights_sf				bundle=gmem4 		depth=SF_DEPTH 				offset=slave max_read_burst_length=16
#pragma HLS INTERFACE mode=m_axi port=value_weights_q					bundle=gmem5 		depth=QUANT_DEPTH 		offset=slave max_read_burst_length=128
/* *************************************************************************	******************************************************/
#pragma HLS INTERFACE mode=m_axi port=mha_output_weights_sf 	bundle=gmem0a 	depth=SF_DEPTH 				offset=slave max_read_burst_length=16		
#pragma HLS INTERFACE mode=m_axi port=mha_output_weights_q 		bundle=gmem1a 	depth=QUANT_DEPTH 		offset=slave max_read_burst_length=256		
/* *************************************************************************	*********/
#pragma HLS INTERFACE mode=m_axi port=rms_ffn_w 							bundle=gmem7b 	depth=TOK_RMS_DEPTH 	offset=slave max_read_burst_length=128	
#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights1_sf 		bundle=gmem2a 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=16	
#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights1_q 			bundle=gmem3a 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=128	
#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights3_sf 		bundle=gmem4a 	depth=HD_SF_DEPTH			offset=slave max_read_burst_length=16	
#pragma HLS INTERFACE mode=m_axi port=mlp_exp_weights3_q 			bundle=gmem5a 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=128	
#pragma HLS INTERFACE mode=m_axi port=swiglu_comp_weights_sf 	bundle=gmem0b 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=16	
#pragma HLS INTERFACE mode=m_axi port=swiglu_comp_weights_q 	bundle=gmem1b 	depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=256	
#pragma HLS INTERFACE mode=m_axi port=wcls_weights_sf 				bundle=gmem0c 	depth=6144 						offset=slave max_read_burst_length=16	
#pragma HLS INTERFACE mode=m_axi port=wcls_weights_q 					bundle=gmem1c 	depth=6144 						offset=slave max_read_burst_length=128	
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

#pragma HLS STREAM variable=first_to_mha_tokens_stream 		depth=48
#pragma HLS STREAM variable=mha_to_final_tokens_stream		depth=48
#pragma HLS STREAM variable=query_stream 									depth=48

#pragma HLS STREAM variable=s_tokens_in depth=2


	/* ******************* QUERY, KEY, VALUE MATRIX MULTIPLICATION KERNEL **********************/

	
	
	// for (int CURR_LAYER = 0; CURR_LAYER < LAYERS; CURR_LAYER++) {
	// 	#pragma HLS LOOP_TRIPCOUNT max = 12
	constexpr int CURR_LAYER = 0;
#pragma HLS DATAFLOW
		tok_load_input(s_tokens_in, tokens, s_tokens_out_loop, CURR_LAYER);

		// rms_load_input(s_rms_att_w, rms_att_w, CURR_LAYER);

		// mm_load_input(s_key_weights_sf, key_weights_sf, SQUARE_SF, CURR_LAYER);
		// mm_tok_load_input(s_key_weights_q, key_weights_q, SQUARE_TOK, CURR_LAYER);

		// mm_load_input(s_value_weights_sf, value_weights_sf, SQUARE_SF, CURR_LAYER);
		// mm_tok_load_input(s_value_weights_q, value_weights_q, SQUARE_TOK, CURR_LAYER);

		// mm_load_input(s_query_weights_sf, query_weights_sf, SQUARE_SF, CURR_LAYER);
		// mm_tok_load_input(s_weights_query_int8, query_weights_q, SQUARE_TOK, CURR_LAYER);
		df_first_kernel(
			rms_att_w,
			key_weights_sf, key_weights_q,
			value_weights_sf, value_weights_q,
			query_weights_sf, query_weights_q, 
			s_rms_att_w,
			s_key_weights_sf, s_key_weights_q,
			s_value_weights_sf, s_value_weights_q,
			s_query_weights_sf, s_query_weights_q, 
			SQUARE_SF, SQUARE_TOK, CURR_LAYER
);


// s_key_cache_out
// s_value_cache_out
// s_key_cache_in
// s_value_cache_in
		

		first_kernel(	s_key_cache_out, s_value_cache_out, // s_key_cache_out,			s_value_cache_out,
									query_stream,					first_to_mha_tokens_stream,
									s_tokens_in,					s_rms_att_w,
									s_query_weights_sf,		s_query_weights_q,
									s_key_weights_sf,			s_key_weights_q,
									s_value_weights_sf,		s_value_weights_q,
									POS, CURR_LAYER);

		// mha_store_output(key_cache, s_key_cache_out, MODEL_ELEMENTS, CURR_LAYER);
		// mha_store_output(value_cache, s_value_cache_out, MODEL_ELEMENTS, CURR_LAYER);
		// hls::fence({s_key_cache_out}, {s_key_cache_in});
		// hls::fence({s_value_cache_out}, {s_key_cache_out});

		// mha_RAW_store_load(key_cache, s_key_cache_in, s_key_cache_out, CURR_LAYER, POS);
		// mha_RAW_store_load(value_cache, s_value_cache_in, s_value_cache_out, CURR_LAYER, POS);

		mha_WAR_store_load(key_cache, s_key_cache_in, s_key_cache_out, CURR_LAYER, POS);
		mha_WAR_store_load(value_cache, s_value_cache_in, s_value_cache_out, CURR_LAYER, POS);

		/* ******************* MULTI HEAD ATTENTION KERNEL **********************/
		// mha_load_input(s_key_cache_in, key_cache, POS, CURR_LAYER);
		// mha_load_input(s_value_cache_in, value_cache, POS, CURR_LAYER);

		mm_load_input(s_mha_output_weights_sf, mha_output_weights_sf, SQUARE_SF, CURR_LAYER);
		mm_tok_load_input(s_mha_output_weights_q, mha_output_weights_q, SQUARE_TOK, CURR_LAYER);

		mha_kernel(	mha_to_final_tokens_stream,	s_key_cache_in,
								s_value_cache_in, 					first_to_mha_tokens_stream, 
								s_mha_output_weights_sf, 		s_mha_output_weights_q, 
								query_stream,								POS,
								CURR_LAYER);

			
		/* ******************* MLP KERNEL **********************/

		rms_load_input(s_rms_ffn_w, rms_ffn_w, CURR_LAYER);

		mm_load_input(s_mlp_exp_weights1_sf, mlp_exp_weights1_sf, RECT_SF, CURR_LAYER);
		mm_tok_load_input(s_mlp_exp_weights1_q, mlp_exp_weights1_q, RECT_TOK, CURR_LAYER);

		mm_load_input(s_mlp_exp_weights3_sf, mlp_exp_weights3_sf, RECT_SF, CURR_LAYER);
		mm_tok_load_input(s_mlp_exp_weights3_q, mlp_exp_weights3_q, RECT_TOK, CURR_LAYER);

		mm_load_input(s_swiglu_comp_weights_sf, swiglu_comp_weights_sf, RECT_SF, CURR_LAYER);
		mm_tok_load_input(s_swiglu_comp_weights_q, swiglu_comp_weights_q, RECT_TOK, CURR_LAYER);

		final_third(	s_tokens_out, 							s_rms_ffn_w, 
									s_mlp_exp_weights1_sf, 			s_mlp_exp_weights1_q, 
									s_swiglu_comp_weights_sf, 	s_swiglu_comp_weights_q, 
									s_mlp_exp_weights3_sf, 			s_mlp_exp_weights3_q, 
									mha_to_final_tokens_stream);


	store_output(tokens, s_tokens_out, MODEL_ELEMENTS, 0);

/*
		if (CURR_LAYER < (LAYERS - 1)) {
			for (int i = 0;  i < (MODEL_ELEMENTS / MAX_FL_ELEM); i++) {
				#pragma HLS PIPELINE II=1	
				s_tokens_out_loop.write(s_tokens_out.read());
			}
		}else {
			for (int i = 0;  i < (MODEL_ELEMENTS / MAX_FL_ELEM); i++) {
				#pragma HLS PIPELINE II=1	
				s_tokens_out_exit.write(s_tokens_out.read());
			}
		}
	// }
	s_mfdata_v_t s_tokens_out_final, s_rms_final_w;
	s_fdata_v_t s_wcls_weights_sf;
	s_idata_v_t s_wcls_weights_q; 

	#pragma HLS STREAM variable=s_tokens_out_final 		depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_rms_final_w 		depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_wcls_weights_sf 		depth=MODEL_TOKENS / (SM_FL_ELEM * MODEL_SCALING_FACTOR)
	#pragma HLS STREAM variable=s_wcls_weights_q 		depth=MODEL_TOKENS / (MAX_QUANT_ELEM)
	#pragma HLS BIND_STORAGE variable=s_tokens_out_final type=fifo impl=lutram
	#pragma HLS BIND_STORAGE variable=s_rms_final_w type=fifo impl=lutram
	// #pragma HLS BIND_STORAGE variable=s_tokens_out_final type=fifo impl=lutram

	mm_load_input(s_wcls_weights_sf, wcls_weights_sf, RECT_SF, 0); //MODEL_TOKENS / (SM_FL_ELEM * MODEL_SCALING_FACTOR), 0);
	mm_tok_load_input(s_wcls_weights_q, wcls_weights_q, RECT_TOK, 0); //MODEL_ELEMENTS * MODEL_TOKENS / (MAX_QUANT_ELEM), 0);
	//need to define the actual size of the weights that come into this part of the kernel. I didn't provide enough data last time.
	
	rms_load_input(s_rms_final_w, rms_final_w, 0);
	logits_kernel(s_tokens_out_final, s_tokens_out_exit, s_rms_final_w, s_wcls_weights_sf, s_wcls_weights_q);
	
	store_output(output_tokens, s_tokens_out_final, MODEL_ELEMENTS, 0);
*/
	
	return;			
}
