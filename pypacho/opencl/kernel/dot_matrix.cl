#include "name_func.h"

#define FUNCNAME NAME(dot_matrix, Type1, Type2) 


__kernel void FUNCNAME(const int M, const int K, 
       const int N, //const int Srows, const int Scols, 
			 const __global Type1* A,
			 const __global Type2* B,
			 __global Out_Type* C
			 //__local Type1* Asub,
			 //__local Type2* Bsub) {
       ) {
  //const int globalRow = get_global_id(0); // Row ID of C (0..M)
  //const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  
  const int globalRow = TS*get_group_id(0) + row;
  const int globalCol = TS*get_group_id(1) + col;


  __local Out_Type Asub[TS][TS];
  __local Out_Type Bsub[TS][TS];

  Out_Type acc = 0;
  const int a_rectifier = globalRow < M;
  const int b_rectifier = globalCol < N;
  
  const int numTiles = ceil((float) K / TS);
  for(int t = 0; t < numTiles; t++) {
    const int tiledCol = TS*t + col;
    const int tiledRow = TS*t + row;

    Asub[row][col] = (Out_Type)(a_rectifier * A[tiledCol + globalRow*K]);
    Bsub[col][row] = (Out_Type)(b_rectifier * B[globalCol + tiledRow*N]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0; k<TS; k++) {
        #ifdef FMA
          acc = fma(Asub[row][k], Bsub[col][k], acc);
        #else
          acc += Asub[row][k] * Bsub[col][k];
        #endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  } 

/*
  Out_Type acc = 0;
  for (int k=0; k<K; k++) {
      acc += A[k+ globalRow*K] * B[k*N + globalCol];
  }
*/
if((globalRow < M) && (globalCol < N))
  C[globalCol + globalRow*N] = acc;
    // Compute a single element (loop over K)
}

#undef FUNCNAME