#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
  union {
    unsigned int u32;
    float        f32;
    } next, expected, current;
  current.f32    = *addr;
  do {
    expected.f32 = current.f32;
    next.f32     = expected.f32 + val;
    current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                  expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}



__kernel void transpose(__global float *a_t, __global float *a, unsigned a_width, unsigned a_height)
{
  int gid = get_global_id(0);
  int row = gid/a_width;
  int col = gid % a_width;
  int idx_a = row + col*(a_height);
  int idx_a_t = col + row*a_width;
  
  a_t[idx_a_t] = a[idx_a];             
}

__kernel void add(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}

__kernel void subtract(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] - b[gid];
}

__kernel void multiply(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] * b[gid];
}

__kernel void scalar_mult(__global float *a, float b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] * b;
}

__kernel void divide(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] / b[gid];
}

__kernel void dot_matrix(__global float *a,__global float *b, __global float *c,
		  unsigned m, unsigned n, unsigned p)
{
  int gid = get_global_id(0);
  c[gid] = 0;
  int rowC = gid/p;
  int colC = gid%p;
  __global float *pA = &a[rowC*n];
  __global float *pB = &b[colC];
  float sum = 0;
  for(int k=0; k<n; k++)
    {
      pB = &b[colC+k*p];
      sum += (*(pA++))*(*pB);
    }
  c[gid] = sum;
}

__kernel void dot_matrix2(const int AROWS, const int ACOLS, const int BROWS, const int BCOLS, const int TS,
                      __global float* A,
                      __global float* B,
                      __global float* C,
                      __local float* Asub,
                      __local float* Bsub) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    //const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    //const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    // Local memory to fit a tile of TS*TS elements of A and B
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = (ACOLS-1)/TS+1;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        if(globalRow < AROWS && tiledCol < ACOLS){
          Asub[col*TS + row] = A[tiledCol + ACOLS*globalRow];
        }
        else{
          Asub[col*TS + row] = 0;
        }
        if(globalCol < BCOLS && tiledRow < BROWS){
          Bsub[col*TS + row] = B[globalCol+ BCOLS*tiledRow];
        }
        else{
          Bsub[col*TS + row] = 0;
        }
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k*TS + row] * Bsub[col*TS + k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    if(globalRow<AROWS && globalCol <BCOLS){
      C[globalCol+ ACOLS * globalRow] = acc;
    }
}


__kernel void vec_dot(__global float* a, __global float* b, __global float* partial_sums,
                          __local float* local_sum, int vec_size) {
  int lid = get_local_id(0);
  uint gid = get_global_id(0);
  uint local_size = get_local_size(0);

  float private_sum = a[gid] * b[gid];

  local_sum[lid] = private_sum;

  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  if(lid == 0)
  {
    int num_groups = get_num_groups(0);
    int group_id = get_group_id(0);
    bool change_size = group_id/num_groups;
    int new_size = vec_size - num_groups*local_size;

    //local size = new_size if change_size == True, local_size otherwise
    local_size = change_size * new_size + (1 - change_size) * local_size;

    for (int i = 0; i < local_size; i++)
    {
      sum += local_sum[i];
    }
    atomicAdd_g_f(&partial_sums[0], sum);
  }
}

__kernel void negative(__global float *a_n, __global float *a)
{
  int gid = get_global_id(0);
  a_n[gid] = -a[gid];             
}

__kernel void sqrt_(__global float *a_n, __global float *a)
{
  int gid = get_global_id(0);
  a_n[gid] = half_sqrt(a[gid]);             
}

__kernel void norm(__global float *norm, __global float *a)
{
  int gid = get_global_id(0);
  norm[0] = half_sqrt(a[gid]);
}

__kernel void diag(__global float *a, __global float *b, int a_size)
{
  int gid = get_global_id(0);
  b[gid] = a[gid*a_size + gid];
}

__kernel void diagflat(__global float *a, __global float *b, int a_size)
{
  int gid = get_global_id(0);
  if(gid < a_size){
    b[gid*a_size + gid] = a[gid];
  }
}


/******************************************************************** */


inline void atomicAdd_g_d(volatile __global double *addr, double val)
{
  union {
    unsigned int u32;
    double        f32;
    } next, expected, current;
  current.f32    = *addr;
  do {
    expected.f32 = current.f32;
    next.f32     = expected.f32 + val;
    current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                  expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

__kernel void double_transpose(__global double *a_t, __global double *a, unsigned a_width, unsigned a_height)
{
  int gid = get_global_id(0);
  int row = gid/a_width;
  int col = gid % a_width;
  int idx_a = row + col*(a_height);
  int idx_a_t = col + row*a_width;
  a_t[idx_a_t] = a[idx_a];             
}

__kernel void double_add(__global double *a, __global double *b, __global double *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}

__kernel void double_subtract(__global double *a, __global double *b, __global double *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] - b[gid];
}

__kernel void double_multiply(__global double *a, __global double *b, __global double *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] * b[gid];
}

__kernel void double_scalar_mult(__global double *a, double b, __global double *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] * b;
}

__kernel void double_divide(__global double *a, __global double *b, __global double *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] / b[gid];
}

__kernel void double_dot_matrix(__global double *a,__global double *b, __global double *c,
		  unsigned m, unsigned n, unsigned p)
{
  int gid = get_global_id(0);
  c[gid] = 0;
  int rowC = gid/p;
  int colC = gid%p;
  __global double *pA = &a[rowC*n];
  __global double *pB = &b[colC];
  double sum = 0;
  for(int k=0; k<n; k++)
    {
      pB = &b[colC+k*p];
      sum += (*(pA++))*(*pB);
    }
  c[gid] = sum;
}


__kernel void double_dot_matrix2(const int AROWS, const int ACOLS, const int BROWS, const int BCOLS, const int TS,
                      __global double* A,
                      __global double* B,
                      __global double* C,
                      __local double* Asub,
                      __local double* Bsub) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    //const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    //const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    // Local memory to fit a tile of TS*TS elements of A and B
 
    // Initialise the accumulation register
    double acc = 0;
    
    // Loop over all tiles
    const int numTiles = (ACOLS-1)/TS+1;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        if(globalRow < AROWS && tiledCol < ACOLS){
          Asub[col*TS + row] = A[tiledCol + ACOLS*globalRow];
        }
        else{
          Asub[col*TS + row] = 0;
        }
        if(globalCol < BCOLS && tiledRow < BROWS){
          Bsub[col*TS + row] = B[globalCol+ BCOLS*tiledRow];
        }
        else{
          Bsub[col*TS + row] = 0;
        }
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k*TS + row] * Bsub[col*TS + k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    if(globalRow<AROWS && globalCol <BCOLS){
      C[globalCol+ ACOLS * globalRow] = acc;
    }
}


__kernel void double_vec_dot(__global double* a, __global double* b, __global double* partial_sums,
                          __local double* local_sum, int vec_size) {
  int lid = get_local_id(0);
  uint gid = get_global_id(0);
  uint local_size = get_local_size(0);

  double private_sum = a[gid] * b[gid];

  local_sum[lid] = private_sum;

  barrier(CLK_LOCAL_MEM_FENCE);
  double sum = 0;
  if(lid == 0)
  {
    int num_groups = get_num_groups(0);
    int group_id = get_group_id(0);
    bool change_size = group_id/num_groups;
    int new_size = vec_size - num_groups*local_size;

    //local size = new_size if change_size == True, local_size otherwise
    local_size = change_size * new_size + (1 - change_size) * local_size;

    for (int i = 0; i < local_size; i++)
    {
      sum += local_sum[i];
    }
    atomicAdd_g_d(&partial_sums[0], sum);
  }
}

__kernel void double_negative(__global double *a_n, __global double *a)
{
  int gid = get_global_id(0);
  a_n[gid] = -a[gid];             
}

__kernel void double_sqrt_(__global double *a_n, __global double *a)
{
  int gid = get_global_id(0);
  a_n[gid] = half_sqrt(a[gid]);             
}

__kernel void double_diag(__global double *a, __global double *b, int a_size)
{
  int gid = get_global_id(0);
  b[gid] = a[gid*a_size + gid];
}

__kernel void double_diagflat(__global double *a, __global double *b, int a_size)
{
  int gid = get_global_id(0);
  if(gid < a_size){
    b[gid*a_size + gid] = a[gid];
  }
}