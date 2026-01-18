
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

	Book analogy:
		===============================
				TOKEN-MAJOR MEMORY LAYOUT
		===============================

 		|----HEAD SIZE -------|\
		|											|	\
		|	'Paperback book'		|	 \
		|		approach					|		\
		H	  									|		 \
		E		Page 1 of 	 			|			\
		A		Max Sequence			|			 \
		D		Length (256)			|			  \
		|											|				 |
		|		12 sentences			|				 |
		|		per 'page'				|				 |
		-----------------------				 |
		\											 \			 |
		 \										  \			 |
			\											 \		 |
			 \			POS							\		 |
				\											 \	 |
				 \											\	 |
					\											 \ |
					 \-----------------------

(volumee 1 of 12) where each volume is a hidden layer
Here , I reach a sentence, (head size), the I turn the page (position)

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

		===============================
				HEAD-MAJOR MEMORY LAYOUT
		===============================

 		|----HEAD SIZE -------|\
		|											|	\
		|		'News paper'			|	 \
		|		Approach					|		\
		P											|		 \
		O		Page 1 of 12			|			\
		S		(Hidden dim)			|			 \
		|											|			  \
		|		256 sentences			|				 |
		|		per 'page					|				 |
		|											|				 |
		-----------------------				 |
		\											 \			 |
		 \										  \			 |
			\											 \		 |
			 \			HEAD						\		 |
				\											 \	 |
				 \											\	 |
					\											 \ |
					 \-----------------------


 Here, I read all the sentences (had size) on the page (position) before turning to the next page (HEAD)
*/

#include "forward.h"
#include "mha.h"

void wide_mha_iterate(hls::stream<my_float_t> &out, s_mfdata_v_t & query, s_mfdata_v_t &key_cache, const int POS){
	
	const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const my_float_t score_scalar = 1.0f / sqrtf((float) MODEL_HEAD_SIZE);
	std::array<mfdata_v_t, (array_size)> query_arr;
	// std::array<my_float_t, (MODEL_SEQUENCE_LEN)> att_arr;
	my_float_t att = 0.0f;
	std::array<my_float_t, (array_size)> score;
	// #pragma HLS ARRAY_PARTITION variable=att_arr complete
	#pragma HLS ARRAY_PARTITION variable=score complete
	#pragma HLS ARRAY_PARTITION variable=query_arr complete

	// mha_loop:
	// for (size_t i = 0; i < MODEL_NUM_HEADS; i++){
		
		//get 64 elements of query
		query_loop:
		for (size_t j = 0; j < array_size; j++){
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
				score[j] = temp.reduce_add();
			}

			for (size_t j = 0; j < array_size; j++){
				#pragma HLS UNROLL
				att += score[j];
			}
			out.write(att * score_scalar);
			att = 0.0f;
		}
	// }
}


void wide_mha_softmax(my_float_t att_out[MODEL_SEQUENCE_LEN], hls::stream<my_float_t> &att_in, const int POS){

	// const int MHA_HEAD_SIZE = N_DIM / NUM_OF_HEADS;
	const size_t unroll_factor = 4;
	// constexpr int ARR_SIZE = SEQ_LEN;
	my_float_t att_arr[MODEL_SEQUENCE_LEN];
	#pragma HLS ARRAY_PARTITION variable=att_arr cyclic factor=4
	// my_float_t temp = 0.0f;


	// softmax_num_of_heads_loop_for_df:
	// for (int h = 0; h < MODEL_NUM_HEADS; h++) {

		my_float_t max_val = std::numeric_limits<float>::lowest();

		att_arr_loop:
		for (size_t i = 0; i < MODEL_SEQUENCE_LEN; i++){
			#pragma HLS PIPELINE II=4
			my_float_t val;
			if (i < POS){
				val = att_in.read();
				if (val > max_val) max_val = val;
			} else{
				val = std::numeric_limits<float>::lowest();
			}
			att_arr[i] = val;
		}
		
		my_float_t final_soft_sum = 0.0f;
		std::array<my_float_t, 4> soft_sum = {0.0f};
		#pragma HLS ARRAY_PARTITION variable=soft_sum complete

		softmax_exp_loop:
		for (size_t i = 0; i < MODEL_SEQUENCE_LEN; i+=unroll_factor ){
			#pragma HLS PIPELINE
			for (size_t j = 0; j < unroll_factor; j++){
				size_t n = i + j;
				my_float_t calc = hls::expf(att_arr[n] - max_val);
				soft_sum[j] += calc;
				att_arr[n] = calc;
			}
		}

		softmax_sum_loop:
		for (size_t i = 0; i < unroll_factor; i++){
			#pragma HLS UNROLL
			final_soft_sum += soft_sum[i];
		}
		
		my_float_t inv_soft_sum = 1.0f/final_soft_sum;

		softmax_normalize_loop:
		for (int i = 0; i < MODEL_SEQUENCE_LEN; i++) {
			// #pragma HLS LOOP_TRIPCOUNT max=(SEQ_LEN + 1) min=1
			#pragma HLS PIPELINE
			att_out[i] = att_arr[i] * inv_soft_sum;
		}
		// max_val = std::numeric_limits<float>::min();
	// }
}


void wide_mha_weighted_sum(s_mfdata_v_t &xb, my_float_t att_in[MODEL_SEQUENCE_LEN], s_mfdata_v_t &value_cache, const int POS){

	constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	mfdata_v_t xb_arr[ARR_SIZE];
	#pragma HLS ARRAY_PARTITION variable=xb_arr complete

	// ws_head_dataflow_loop:
	// for (size_t h = 0; h < MODEL_NUM_HEADS; h++){

		mha_ws_zero_xb_loop: // set all values to zero
		for (int i = 0 ; i < ARR_SIZE; i++) {
			#pragma HLS UNROLL
			xb_arr[i] = 0.0f;
		}

		mha_pos_loop:
		for (size_t t = 0; t < POS; t++){
			#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
			#pragma HLS PIPELINE II=8

			for (size_t i = 0; i < ARR_SIZE; i++){
				#pragma HLS UNROLL
				xb_arr[i] += att_in[t] * value_cache.read();
			}
		}
		mha_ws_stream_out_xb: // set all values to zero
		for (int i = 0 ; i < ARR_SIZE; i++) {
			// #pragma HLS PIPELINE II
			xb.write(xb_arr[i]);
		}
	// }
}

void wide_mha_kernel(s_mfdata_v_t &xb, 
								s_mfdata_v_t &key_cache,
								s_mfdata_v_t &value_cache,
								s_mfdata_v_t &query,
								const int POS){

	//
	#pragma HLS DATAFLOW
	my_float_t att_sm_ws[MODEL_SEQUENCE_LEN];
	hls::stream<my_float_t> mha_it_sm;
	#pragma HLS STREAM variable=mha_it_sm depth=8

	wide_mha_iterate(mha_it_sm, query, key_cache, POS);

	wide_mha_softmax(att_sm_ws, mha_it_sm, POS);

	wide_mha_weighted_sum(xb, att_sm_ws, value_cache, POS);
}