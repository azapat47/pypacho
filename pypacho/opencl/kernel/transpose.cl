#include "name_func.h"

#define FUNCNAME NAME1(transpose, Type1)

__kernel void FUNCNAME(__global Type1 *a_t, const __global Type1 *a,
                       __local Type1 *sub, unsigned sub_height, unsigned sub_width,
                       unsigned a_height, unsigned a_width) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if((row < a_width) && (col < a_height)) {
    unsigned int index_in = col * a_width + row;
		sub[get_local_id(1)*(sub_height+1)+get_local_id(0)] = a[index_in];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  row = get_group_id(1) * sub_width + get_local_id(0);
	col = get_group_id(0) * sub_width + get_local_id(1);

  if((row < a_height) && (col < a_width)) {
    unsigned int index_out = col * a_height + row;
		a_t[index_out] = sub[get_local_id(0)*(sub_height+1)+get_local_id(1)];
  }
}

#undef FUNCNAME
