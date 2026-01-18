#include "forward.h"
#include "matmul.h"
#include "rmsnorm.h"
#include "quantizer.h"

void final_kernel_seq_a(
	s_mfdata_v_t &tokens_out,
	s_fdata_v_t &tok_sf, s_idata_v_t &tok_q,
	mfdata_v_t *rms_ffn_w, s_mfdata_v_t &tokens_in, 
	const int CURR_LAYER){
	
	s_mfdata_v_t s_rms_ffn_w("rms feed forward network weights");
	s_mfdata_v_t s_tokens_rms_ffn("Tokens Input to RMSNorm");
	s_mfdata_v_t s_rms_ffn_to_quant("RMS Norm to Quantizer kernel");

  #pragma HLS STABLE variable=CURR_LAYER
	#pragma HLS STREAM variable=s_rms_ffn_w depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_tokens_rms_ffn depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_rms_ffn_to_quant depth=(MODEL_ELEMENTS / MAX_FL_ELEM)

	#pragma HLS INLINE

	split_tee(s_tokens_rms_ffn, tokens_out, tokens_in, (MODEL_ELEMENTS/MAX_FL_ELEM));
	rms_quant_router(tok_sf, tok_q, rms_ffn_w, s_tokens_rms_ffn, CURR_LAYER);
	}

void final_third_dataflow(
	s_fdata_v_t &s_mlp_exp_weights1_sf, s_idata_v_t &s_mlp_exp_weights1_q,
	s_fdata_v_t &s_mlp_exp_weights3_sf, s_idata_v_t &s_mlp_exp_weights3_q,
	s_fdata_v_t &s_swiglu_comp_weights_sf, s_idata_v_t &s_swiglu_comp_weights_q,
	fdata_v_t *mlp_exp_weights1_sf, idata_v_t *mlp_exp_weights1_q,
	fdata_v_t *mlp_exp_weights3_sf, idata_v_t *mlp_exp_weights3_q,
	fdata_v_t *swiglu_comp_weights_sf, idata_v_t *swiglu_comp_weights_q,
	const int CURR_LAYER){

	#pragma HLS STABLE variable=CURR_LAYER	

	#pragma HLS DATAFLOW

	mm_load_input(s_mlp_exp_weights1_sf, mlp_exp_weights1_sf, RECT_SF, CURR_LAYER);
	mm_tok_load_input(s_mlp_exp_weights1_q, mlp_exp_weights1_q, RECT_TOK, CURR_LAYER);

	mm_load_input(s_mlp_exp_weights3_sf, mlp_exp_weights3_sf, RECT_SF, CURR_LAYER);
	mm_tok_load_input(s_mlp_exp_weights3_q, mlp_exp_weights3_q, RECT_TOK, CURR_LAYER);
	
	mm_load_input(s_swiglu_comp_weights_sf, swiglu_comp_weights_sf, RECT_SF, CURR_LAYER);
	mm_tok_load_input(s_swiglu_comp_weights_q, swiglu_comp_weights_q, RECT_TOK, CURR_LAYER);
}

void final_third_seq_b(
	s_mfdata_v_t &output_tokens,
	s_mfdata_v_t &input_tokens,
	fdata_v_t *mlp_exp_weights1_sf, idata_v_t *mlp_exp_weights1_q,
	fdata_v_t *mlp_exp_weights3_sf, idata_v_t *mlp_exp_weights3_q,
	fdata_v_t *swiglu_comp_weights_sf, idata_v_t *swiglu_comp_weights_q,
	mfdata_v_t *rms_ffn_w,	const int CURR_LAYER){

	#pragma HLS STABLE variable=CURR_LAYER
	#pragma HLS INLINE
	
	s_fdata_v_t quant_mm_sf("From swiglu to MLP 2 Matrix mult SF");
	s_idata_v_t quant_mm_q("From swiglu to MLP 2 Matrix mult quant");
	s_mfdata_v_t mm_rc_out("mm rc out");
	s_fdata_v_t tok_sf;
	s_idata_v_t tok_q;

	// s_mfdata_v_t spl_tok;
	s_mfdata_v_t s_swi_w1, s_swi_w3, hb_out, spl_tok;
	s_mfdata_v_t s_rms_ffn_w("rms feed forward network weights");
	s_mfdata_v_t s_tokens_rms_ffn("Tokens Input to RMSNorm");
	s_mfdata_v_t s_rms_ffn_to_quant("RMS Norm to Quantizer kernel");
	#pragma HLS STREAM variable=s_rms_ffn_w depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_tokens_rms_ffn depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_rms_ffn_to_quant depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_swi_w1 depth=(MODEL_HIDDEN_DIM / MAX_FL_ELEM)
	#pragma HLS STREAM variable=s_swi_w3 depth=(MODEL_HIDDEN_DIM / MAX_FL_ELEM)
	#pragma HLS STREAM variable=hb_out depth=(MODEL_HIDDEN_DIM / MAX_FL_ELEM)
	#pragma HLS STREAM variable=spl_tok depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=mm_rc_out depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=tok_sf depth=(MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR))
	#pragma HLS STREAM variable=tok_q depth=(MODEL_ELEMENTS / MAX_FL_ELEM)
	#pragma HLS STREAM variable=quant_mm_sf depth=(MODEL_HIDDEN_DIM / (SM_FL_ELEM * MODEL_SCALING_FACTOR))
	#pragma HLS STREAM variable=quant_mm_q depth=(MODEL_HIDDEN_DIM / MAX_FL_ELEM)
	
	split_tee(s_tokens_rms_ffn, spl_tok, input_tokens, (MODEL_ELEMENTS/MAX_FL_ELEM));
	rms_quant_router(tok_sf, tok_q, rms_ffn_w, s_tokens_rms_ffn, CURR_LAYER);
	
	two_matmult_kernel(s_swi_w1, s_swi_w3, tok_sf, tok_q, mlp_exp_weights1_sf, mlp_exp_weights1_q, mlp_exp_weights3_sf, mlp_exp_weights3_q, MODEL_ELEMENTS, MODEL_HIDDEN_DIM, CURR_LAYER);
	swiglu_kernel<MODEL_HIDDEN_DIM>(hb_out, s_swi_w1, s_swi_w3);

	quantizer_kernel<MODEL_HIDDEN_DIM>(quant_mm_sf, quant_mm_q, hb_out);
	
	// final_two_matmult_kernel(quant_mm_sf, quant_mm_q, tok_sf, tok_q, mlp_exp_weights1_sf, mlp_exp_weights1_q, mlp_exp_weights3_sf, mlp_exp_weights3_q, MODEL_ELEMENTS, MODEL_HIDDEN_DIM, CURR_LAYER);
	
	// matmult_kernel(mm_rc_out, quant_mm_sf, quant_mm_q, swiglu_comp_weights_sf, s_swiglu_comp_weights_q, MODEL_HIDDEN_DIM, MODEL_ELEMENTS);
	one_matmult_kernel(mm_rc_out, quant_mm_sf, quant_mm_q, swiglu_comp_weights_sf, swiglu_comp_weights_q, MODEL_HIDDEN_DIM, MODEL_ELEMENTS, CURR_LAYER);
	resid_conn<MODEL_ELEMENTS>(output_tokens, spl_tok, mm_rc_out);
}