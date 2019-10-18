#include "name_func.h"

#define FUNCNAME NAME(add, Type1, Type2)

__kernel void FUNCNAME(const __global Type1 *a, const __global Type2 *b, __global Out_Type *c,
                       //__local Type1 *sub, unsigned sub_height, unsigned sub_width,
                       unsigned height, unsigned width)
{

  unsigned int row = get_global_id(0);
  unsigned int col = get_global_id(1);

  __local Type1 sub[TS + 1][TS];

  unsigned int index_in = row * width + col;
  if((col < width) && (row < height)) {
	sub[get_local_id(1)][get_local_id(0)] = a[index_in] + b[index_in];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if((row < height) && (col < width)) {
  	c[index_in] = sub[get_local_id(1)][get_local_id(0)];
  }
  //int gid = get_global_id(0);
  //c[gid] = a[gid] + b[gid];
}

#undef FUNCNAME