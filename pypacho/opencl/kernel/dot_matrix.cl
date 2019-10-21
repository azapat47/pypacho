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

  Out_Type Areg;
  Out_Type Breg[WPT];

  __local Out_Type Asub[TS][TS];
  __local Out_Type Bsub[TS][TS];

  Out_Type acc[WPT][WPT];
  #pragma unroll
  for(int wi = 0; wi < WPT; wi++) {
      #pragma unroll
      for(int wj = 0; wj < WPT; wj++) {
        acc[wi][wj] = 0;
      }
  }
  
  const int numTiles = ceil((float) K / TS);
  for(int t = 0; t < numTiles; t++) {
    const int tiledCol = TS*t + col;
    const int tiledRow = TS*t + row;

    #pragma unroll
    for(int wi= 0; wi < WPT; wi++) {
      const bool a_rectifier = globalRow + wi*RTS < M;
      #pragma unroll
      for(int wj = 0; wj < WPT; wj++) {
        const int b_rectifier = globalCol + wj*RTS < N;
        Asub[row + wi*RTS][col + wj*RTS] = (Out_Type)(a_rectifier * A[tiledCol + wj*RTS + (globalRow + wi*RTS)*K]);
        Bsub[col + wj*RTS][row + wi*RTS] = (Out_Type)(b_rectifier * B[globalCol + wj*RTS + (tiledRow + wi*RTS)*N]);
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for(int k=0; k<TS; k++) {

        #pragma unroll
        for(int w = 0; w < WPT; w++) {
          Breg[w] = Bsub[col + w*RTS][k];
        }

        #pragma unroll
        for(int wi = 0; wi < WPT; wi++) {
          Areg = Asub[row + wi*RTS][k];
          #pragma unroll
          for(int wj = 0; wj < WPT; wj++) {
            #ifdef FMA
              acc[wi][wj] = fma(Areg, Breg[wj], acc[wi][wj]);
            #else
              acc[wi][wj] += Areg * Breg[wj];
            #endif
          }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  } 

  #pragma unroll
  for(int wi = 0; wi < WPT; wi++) {
    const int row_index = (globalRow + wi*RTS);
    #pragma unroll
    for(int wj = 0; wj < WPT; wj++) {
      if((row_index < M) && (globalCol + wj*RTS < N))
        C[globalCol + wj*RTS + row_index*N] = acc[wi][wj];
    }
  }
}

#undef FUNCNAME