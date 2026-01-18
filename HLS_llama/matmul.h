#ifndef MARK_MATMUL
#define MARK_MATMUL

#include "forward.h"
void matmult_kernel(s_mfdata_v_t &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf, s_idata_v_t &w, const int N_DIM, const int M_DIM);
// void two_matmult_kernel(s_mfdata_v_t &out1, s_mfdata_v_t &out2, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf1, s_idata_v_t &w1, s_fdata_v_t &w_sf2, s_idata_v_t &w2);
// void two_matmult_kernel(s_mfdata_v_t &out1, s_mfdata_v_t &out2, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf1, s_idata_v_t &w1, s_fdata_v_t &w_sf2, s_idata_v_t &w2, const int N_DIM, const int M_DIM);
void two_matmult_kernel(s_mfdata_v_t &out1, s_mfdata_v_t &out2, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf1, idata_v_t *w1, fdata_v_t *w_sf2, idata_v_t *w2, const int N_DIM, const int M_DIM, const int CURR_LAYER);
void one_matmult_kernel(s_mfdata_v_t &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER);
void rope_kernel (s_mfdata_v_t &out, s_mfdata_v_t &in, const int N_DIM, const int POS);
void two_matmult_sg_kernel(/*s_mfdata_v_t &hb_out,*/ s_fdata_v_t &hb_sf, s_idata_v_t &hb_tok, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf1, s_idata_v_t &w1, s_fdata_v_t &w_sf2, s_idata_v_t &w2);
void rms_quant_router( s_fdata_v_t &quant_sf, s_idata_v_t &quant_tok, mfdata_v_t *rms_w, s_mfdata_v_t &in_tokens, const int CURR_LAYER);
void final_two_matmult_kernel(s_fdata_v_t &hb_sf, s_idata_v_t &hb_tok, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf1, idata_v_t *w1, fdata_v_t *w_sf2, idata_v_t *w2, const int N_DIM, const int M_DIM, const int CURR_LAYER);
void logits_matmult_kernel(mfdata_v_t *out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER);

// void fix_matmult_kernel(s_mfdata_v_t &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok, s_fdata_v_t &w_sf, s_idata_v_t &w, const int N_DIM, const int M_DIM);
#endif