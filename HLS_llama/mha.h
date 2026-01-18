
#ifndef MARK_MHA_H__
#define MARK_MHA_H__

#include "forward.h"

template<size_t HEAD_SIZE>
void wide_mha_kernel(s_mfdata_v_t &xb, 
								s_mfdata_v_t &key_cache,
								s_mfdata_v_t &value_cache,
								s_mfdata_v_t &query,
								const int POS);
#endif