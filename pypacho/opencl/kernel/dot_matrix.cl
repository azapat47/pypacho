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

  Out_Type acc[WPT];
  for(int w = 0; w < WPT; w++) {
    acc[w] = 0;
  }

  const int a_rectifier = globalRow < M;
  
  const int numTiles = ceil((float) K / TS);
  for(int t = 0; t < numTiles; t++) {
    const int tiledCol = TS*t + col;
    const int tiledRow = TS*t + row;

    for(int w = 0; w < WPT; w++) {
     const int b_rectifier = globalCol + w*RTS < N;
     Asub[row][col + w*RTS] = (Out_Type)(a_rectifier * A[tiledCol + w*RTS + globalRow*K]);
     Bsub[col + w*RTS][row] = (Out_Type)(b_rectifier * B[globalCol + w*RTS + tiledRow*N]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0; k<TS; k++) {
        for(int w = 0; w < WPT; w++) {
          #ifdef FMA
            acc[w] = fma(Asub[row][k], Bsub[col + w*RTS][k], acc[w]);
          #else
            acc[w] += Asub[row][k] * Bsub[col + w*RTS][k];
          #endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  } 

/*
  Out_Type acc = 0;
  for (int k=0; k<K; k++) {
      acc += A[k+ globalRow*K] * B[k*N + globalCol];
  }
*/
  for(int w = 0; w < WPT; w++) {
    if((globalRow < M) && (globalCol + w*RTS < N))
      C[globalCol + w*RTS + globalRow*N] = acc[w];
        // Compute a single element (loop over K)
  }
}

#undef FUNCNAME