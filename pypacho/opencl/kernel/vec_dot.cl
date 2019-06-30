#include "name_func.h"

#define FUNCNAME NAME(vec_dot, Type1, Type2) 

__kernel void FUNCNAME(const __global Type1 *a, const __global Type2 *b, 
                       __global Out_Type *c,
		               __local Out_Type *local_sum, const int vec_size) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);
  const int local_size = get_local_size(0);

  if(gid < vec_size) {
    local_sum[lid] = a[gid] * b[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0) {
      Out_Type private_sum = 0;

      const int num_groups = get_num_groups(0) - 1;
      const int group_id = get_group_id(0);
      const bool change_size = group_id/num_groups;
      int new_size = vec_size - num_groups*local_size;

      //local size = new_size if change_size == True, local_size otherwise
      new_size = change_size * new_size + (1 - change_size) * local_size;
      
      for(int i = 0; i < new_size; i++) {
	    private_sum += local_sum[i];
      }
      Atomic_Add(&c[0], private_sum);
    }
  }
}
#undef FUNCNAME