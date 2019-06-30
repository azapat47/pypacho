#include "name_func.h"

#define FUNCNAME NAME(matrix_vec, Type1, Type2) 

__kernel void FUNCNAME(const __global Type1 *A, const __global Type2 *vec,
                       __global Out_Type *c,
			                 __local Out_Type *Asub, __local Type2 *vecsub,
			                 const int vec_size, const int matrix_rows) {

  const int globalCol = get_global_id(1);
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int local_size_rows = get_local_size(0);
  const int local_size_cols = get_local_size(1);

  if(globalCol < vec_size) {
    if(row == 0) {
      vecsub[col] = vec[globalCol];
    }

    const int num_tiles = ceil( (float) matrix_rows / local_size_rows);
    for(int t = 0; t < num_tiles; t++) {
      barrier(CLK_LOCAL_MEM_FENCE);
    
      const int globalRow = t*local_size_cols + row;
      if(globalRow < matrix_rows) {
	      Asub[row*local_size_cols + col] = A[globalRow*vec_size + globalCol] * vecsub[col];
	      barrier(CLK_LOCAL_MEM_FENCE);

	      if(col == 0) {
	        Out_Type private_sum = 0;
	        const int num_groups = get_num_groups(1) - 1;
	        const int group_id = get_group_id(1);
	        const bool change_size = group_id/num_groups;
	        int new_size = vec_size - num_groups*local_size_cols;

	        //local size = new_size if change_size == True, local_size otherwise
	        new_size = change_size * new_size + (1 - change_size) * local_size_cols;
	        for(int i = 0; i < new_size; i++) {
	          private_sum += Asub[row*local_size_cols + i];
	        }
	        Atomic_Add(&c[globalRow], private_sum);
	      }
      }
    }
  }
}
#undef FUNCNAME