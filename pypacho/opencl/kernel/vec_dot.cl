#include "name_func.h"

#define FUNCNAME NAME(vec_dot, Type1, Type2) 

__kernel void FUNCNAME(const __global Type1 *a, const __global Type2 *b, 
                       __global Out_Type *c, const int vec_size) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);

  const int local_size = get_local_size(0);
  __local Out_Type local_sum[FS];

  const int rectifier = gid < vec_size;
  local_sum[lid] = rectifier * a[gid] * b[gid];

  barrier(CLK_LOCAL_MEM_FENCE);

  if(lid == 0) {
    Out_Type private_sum = 0;

    for(int i = 0; i < FS; i++) {
      private_sum += local_sum[i];
    }
    Atomic_Add(&c[0], private_sum);
  }
}
#undef FUNCNAME