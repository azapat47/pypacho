#include "name_func.h"

#define FUNCNAME NAME(matrix_vec, Type1, Type2) 

__kernel void FUNCNAME(const __global Type1 *A, const __global Type2 *vec,
                       __global Out_Type *c,
			                 const int vec_size, const int matrix_rows) {

  const int globalCol = get_global_id(1);
  const int row = get_local_id(0);
  const int col = get_local_id(1);

  __local Out_Type Asub[TS][TS];
  __local Out_Type vecsub[TS];

  if(row == 0) {
    vecsub[col] = vec[globalCol];
  }

  const int num_tiles = ceil( (float) matrix_rows / TS);
  const int rectifier = globalCol < vec_size;
  for(int t = 0; t < num_tiles; t++) {
    barrier(CLK_LOCAL_MEM_FENCE);
  
    const int globalRow = t*TS + row;
    if(globalRow < matrix_rows) {
      Asub[row][col] = rectifier * A[globalRow*vec_size + globalCol] * vecsub[col];
      barrier(CLK_LOCAL_MEM_FENCE);

      if(col == 0) {
        Out_Type private_sum = 0;
        for(int i = 0; i < TS; i++) {
          private_sum += Asub[row][i];
        }
        Atomic_Add(&c[globalRow], private_sum);
      }
    }
  }
}
#undef FUNCNAME