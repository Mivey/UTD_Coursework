#include "forward.h"
#include "matmul.h"
// #include "mha.h"
#include "quantizer.h"
// #include <etc/hls_cordic_apfixed.h>
#include <hls_math.h>



/*
In the orignial design, Karpathy used a 'Token-Major' memory layout. 
This means we read one head (64 elements) from each position (1 to max seq. length).
The major advantage of this approach is
- Write a 'page' easily. But KV output is relatively small (768 * 4 bytes) vs Read (768 * 256 * 4)
- 'Fast' on first few tokens
This is not good for a few reasons:
- If I want to do dataflow (do one head at a time) I must jump around, up to 1024 times
- If I want to enable burst reads, I can not move to softmax until I have all the calulations in mha_iterate done.
	- may be able to do 12 softmax at the same time, but SM is not the issue
	- means one long key cache read, then one long value cache read


	
The better way is to do a 'Head-Major' memory layout. 
Here we append a new sentence to each page. While this means writes become strided,
our reads are now linear bursts.
Advantages:
- Burst read: 
	- Each 'sentence' on the 'page' is for the given head.
- I can now dataflow (have iterate, softmax and ws all working) and overlap the MHA calculations
- Enables burst read from both Key and Value cache
Disadvantge:
- Write to caches are strided, but constant O(1)


	Book analogy:
		===============================								===============================				
				TOKEN-MAJOR MEMORY LAYOUT											HEAD-MAJOR MEMORY LAYOUT					
		===============================								===============================				
																																												
 		|----HEAD SIZE -------|\									 		|----HEAD SIZE -------|\							
		|											|	\											|											|	\							
		|	'Paperback book'		|	 \										|		'News paper'			|	 \						
		|		approach					|		\										|		Approach					|		\						
		H	  									|		 \									P											|		 \					
		E		Page 1 of 	 			|			\									O		Page 1 of 12			|			\					
		A		Max Sequence			|			 \								S		(Hidden dim)			|			 \				
		D		Length (256)			|			  \								|											|			  \				
		|											|				 |							|		256 sentences			|				 |			
		|		12 sentences			|				 |							|		per 'page					|				 |			
		|		per 'page'				|				 |							|											|				 |			
		-----------------------				 |							-----------------------				 |			
		\											 \			 |							\											 \			 |			
		 \										  \			 |							 \										  \			 |			
			\											 \		 |								\											 \		 |			
			 \			POS							\		 |								 \			HEAD						\		 |			
				\											 \	 |									\											 \	 |			
				 \											\	 |									 \											\	 |			
					\											 \ |										\											 \ |			
					 \-----------------------											 \-----------------------				

(volumee 1 of 12) where each volume is a hidden layer
TOKEN MAJOR:	I read a sentence, (head size), the I turn the page (position)
HEAD MAJOR:		I read all the sentences (head size) on the page (position) before turning to the next page (HEAD)
*/
void wide_mha_iterate(hls::stream<my_float_t> &out, s_mfdata_v_t & query, s_mfdata_v_t &key_cache, const int POS){
	
	const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const my_float_t score_scalar = 1.0f / sqrtf((float) MODEL_HEAD_SIZE);
	std::array<mfdata_v_t, (array_size)> query_arr;
	my_float_t att = 0.0f;
	std::array<my_float_t, (array_size)> score;
	#pragma HLS ARRAY_PARTITION variable=score complete
	#pragma HLS ARRAY_PARTITION variable=query_arr complete
	
	//get 64 elements of query
	query_loop:
	for (size_t j = 0; j < array_size; j++){
		#pragma HLS PIPELINE II=1
		query_arr[j] = query.read();
	}

	pos_loop:
	for (size_t k = 0; k < POS; k++){
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		//att_array adder tree
		att_loop:
		for (size_t j = 0; j < array_size; j++){
			
			#pragma HLS PIPELINE II=4
			mfdata_v_t temp = query_arr[j] * key_cache.read();
			for (int n = 0; n < MAX_FL_ELEM; n++) {
				att += temp[n];
			}
		}
		out.write(att * score_scalar);
		att = 0.0f;
	}
}


void wide_mha_softmax(hls::stream<my_float_t> &att_out, hls::stream<my_float_t> &att_in, const int POS){

	my_float_t att_arr[MODEL_SEQUENCE_LEN] = {std::numeric_limits<float>::lowest()};
	#pragma HLS ARRAY_PARTITION variable=att_arr cyclic factor=4

	my_float_t max_val = std::numeric_limits<float>::lowest();

	sm_new_intake_loop:
	for (int i = 0; i < POS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE II=2
		
		my_float_t val = att_in.read();

		if (max_val < val) {
			max_val = val;
		}
		att_arr[i] = val;
	}
	my_float_t final_soft_sum = 0.0f;
	
	softmax_exp_loop:
	for (int i = 0; i < POS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE
		my_float_t calc = hls::expf((att_arr[i] - max_val));
		final_soft_sum += calc;
		att_arr[i] = calc;
	}
	my_float_t inv_soft_sum = 1.0f/final_soft_sum;

	softmax_normalize_loop:
	for (int i = 0; i < POS; i++) {
		// #pragma HLS LOOP_TRIPCOUNT max=(SEQ_LEN + 1) min=1
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN
		#pragma HLS PIPELINE
		my_float_t tempa = att_arr[i] * inv_soft_sum;
		att_out.write(tempa) ;
	}
}


void wide_mha_weighted_sum(s_mfdata_v_t &xb, hls::stream<my_float_t>  &att_in, s_mfdata_v_t &value_cache, const int POS){

	constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	mfdata_v_t xb_arr[ARR_SIZE] = {0.0f};
	my_float_t att_arr[MODEL_SEQUENCE_LEN] = {0.0f};
	#pragma HLS ARRAY_PARTITION variable=xb_arr complete

	mha_pos_loop:
	for (size_t t = 0; t < POS; t++){
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
		// #pragma HLS PIPELINE II=8
		my_float_t val = att_in.read();
		for (size_t i = 0; i < ARR_SIZE; i++){
			#pragma HLS PIPELINE
			xb_arr[i] += /*att_arr[t]*/ val * value_cache.read();
		}
	}
	mha_ws_stream_out_xb: // set all values to zero
	for (int i = 0 ; i < ARR_SIZE; i++) {
		#pragma HLS PIPELINE II=1
		xb.write(xb_arr[i]);
	}
}

template<size_t HEAD_SIZE>
void wide_mha_kernel(s_mfdata_v_t &xb, 
								s_mfdata_v_t &key_cache,
								s_mfdata_v_t &value_cache,
								s_mfdata_v_t &query,
								const int POS){
	mha_num_head_loop:
	for (size_t i = 0; i < MODEL_NUM_HEADS; i++) {
		// df_wide_mha_kernel<HEAD_SIZE>(xb, key_cache, value_cache, query, POS);
		#pragma HLS DATAFLOW
		
		hls::stream<my_float_t> mha_it_sm, att_sm_ws;
		#pragma HLS STREAM variable=mha_it_sm depth=8

		wide_mha_iterate(mha_it_sm, query, key_cache, POS);
		wide_mha_softmax(att_sm_ws, mha_it_sm, POS);
		wide_mha_weighted_sum(xb, att_sm_ws, value_cache, POS);
	}
}

void mha_kernel_dataflow(s_mfdata_v_t &tokens_out, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                s_mfdata_v_t &tokens_in,
								s_mfdata_v_t &key_cache_in, s_mfdata_v_t &value_cache_in,
                fdata_v_t *mha_output_weights_sf, 
                idata_v_t *mha_output_weights_q, 
                s_mfdata_v_t &query, 
                const int POS, const int CURR_LAYER){
  
	s_fdata_v_t out_weights_sf_stream;
	s_idata_v_t out_weights_q_stream;

	s_mfdata_v_t xb_ws_q("WS to Quantizer for XB Stream");
	s_fdata_v_t xb_sf_q_mm("Quantizer to MM xb_sf stream");
	s_idata_v_t xb_tok_q_mm("Quantizer to MM xb_tok stream");
	s_mfdata_v_t xb_mm_to_rc("xb From mm to Residiual Connection");
	s_mfdata_v_t s_key_cache_to_kernel("From DDR to kernel key cache");
	s_mfdata_v_t s_value_cache_to_kernel("From DDR to kernel value cache");
	s_fdata_v_t s_mha_output_weights_sf("Load MHA Output Weights SF");
	s_idata_v_t s_mha_output_weights_q("Load MHA Output Weights quant");

  #pragma HLS STABLE variable=POS
  #pragma HLS STABLE variable=CURR_LAYER
  #pragma HLS STREAM variable=xb_ws_q depth=2
  #pragma HLS STREAM variable=xb_sf_q_mm depth=2 // probably fine for now, need to re-write mm and quant
  #pragma HLS STREAM variable=xb_tok_q_mm depth=2
  #pragma HLS STREAM variable=out_weights_sf_stream depth=2
  #pragma HLS STREAM variable=out_weights_q_stream depth=2
  #pragma HLS STREAM variable=query depth=32
	#pragma HLS STREAM variable=xb_mm_to_rc depth=4

	#pragma HLS DATAFLOW

	mha_WAR_store_load(key_cache, s_key_cache_to_kernel, key_cache_in, CURR_LAYER, POS);
	mha_WAR_store_load(value_cache, s_value_cache_to_kernel, value_cache_in, CURR_LAYER, POS);
	
	wide_mha_kernel<MODEL_HEAD_SIZE>(xb_ws_q, s_key_cache_to_kernel, s_value_cache_to_kernel, query, POS + 1);

	mm_load_input(s_mha_output_weights_sf, mha_output_weights_sf, SQUARE_SF, CURR_LAYER);
	mm_tok_load_input(s_mha_output_weights_q, mha_output_weights_q, SQUARE_TOK, CURR_LAYER);

  quantizer_kernel<MODEL_ELEMENTS>(xb_sf_q_mm, xb_tok_q_mm, xb_ws_q);

	matmult_kernel(xb_mm_to_rc, xb_sf_q_mm, xb_tok_q_mm, s_mha_output_weights_sf, s_mha_output_weights_q, MODEL_ELEMENTS, MODEL_ELEMENTS);
  resid_conn<MODEL_ELEMENTS>(tokens_out, tokens_in, xb_mm_to_rc);

}
