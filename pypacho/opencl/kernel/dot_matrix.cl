#include "name_func.h"

#define FUNCNAME NAME(dot_matrix, Type1, Type2) 


__kernel void FUNCNAME(const int M, const int N, 
       const int K, const int Srows, const int Scols, 
			 const __global Type1* A,
			 const __global Type2* B,
			 __global Out_Type* C,
			 __local Type1* Asub,
			 __local Type2* Bsub) {
    
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int globalRow = Srows*get_group_id(0) + row;
  const int globalCol = Scols*get_group_id(1) + col;
  Out_Type acc = 0;
  
  const int num_tiles = ceil((float) N / Scols);
  for(int t = 0; t < num_tiles; t++) {
    const int tiledCol = Scols*t + col;
    const int tiledRow = Srows*t + row;
    Asub[row*Scols + col] = A[(globalRow*N + tiledCol)]; 
    Bsub[row*Scols + col] = B[(tiledRow*K + globalCol)];
    barrier(CLK_LOCAL_MEM_FENCE);

    const bool change_size = (t+1) == num_tiles;
    int new_size = N - (num_tiles - 1)*Scols;

    //local size = new_size if change_size == True, local_size otherwise
    new_size = change_size * new_size + (1 - change_size) * Scols;

    for(int i = 0; i < new_size; i++) {
      acc += Asub[row*Scols + i] * Bsub[i*Scols + col];
    }
  }
  if((globalRow < M) && (globalCol < K)){
    C[globalRow*K + globalCol] = acc;
  }
}

#undef FUNCNAME