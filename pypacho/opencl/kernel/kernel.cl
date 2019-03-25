#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

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

__kernel void dot_matrix(const int M, const int K, const int N, const int TS,
                      __global float* A,
                      __global float* B,
                      __global float* C,
                      __local float* Asub,
                      __local float* Bsub) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = ceil((float) K/TS);
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col + row*TS] = A[globalRow*K + t*TS + col];
        Bsub[col + row*TS] = B[N*(t*TS + row) + globalCol];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        const bool change_size = (t+1)/numTiles;
        int new_size = K - (numTiles - 1)*TS;

        //local size = new_size if change_size == True, local_size otherwise
        new_size = change_size * new_size + (1 - change_size) * TS;

        // Perform the computation for a single tile
        for (int k=0; k<new_size; k++) {
            acc = fma(Asub[k + row*TS], Bsub[col + k*TS], acc);
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    if(globalCol < N && globalRow < M)
      C[globalCol + M*globalRow] = acc;
}


__kernel void matrix_vec( __global float* A, __global float* vec, __global float* c,
			  __local float* Asub, __local float* vecsub,
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
	  float private_sum = 0.0f;
	  const int num_groups = get_num_groups(1) - 1;
	  const int group_id = get_group_id(1);
	  const bool change_size = group_id/num_groups;
	  int new_size = vec_size - num_groups*local_size_cols;

	  //local size = new_size if change_size == True, local_size otherwise
	  new_size = change_size * new_size + (1 - change_size) * local_size_cols;
	  for(int i = 0; i < new_size; i++) {
	    private_sum += Asub[row*local_size_cols + i];
	  }
	  atomicAdd_g_f(&c[globalRow], private_sum);
	}
      }
    }
  }
}


__kernel void vec_dot(__global float* a, __global float* b, __global float* c,
		      __local float* local_sum, int vec_size) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);
  const int local_size = get_local_size(0);

  if(gid < vec_size) {
    local_sum[lid] = a[gid] * b[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0) {
      float private_sum = 0.0f;

      const int num_groups = get_num_groups(0) - 1;
      const int group_id = get_group_id(0);
      const bool change_size = group_id/num_groups;
      int new_size = vec_size - num_groups*local_size;

      //local size = new_size if change_size == True, local_size otherwise
      new_size = change_size * new_size + (1 - change_size) * local_size;
      
      for(int i = 0; i < new_size; i++) {
	private_sum += local_sum[i];
      }
      atomicAdd_g_f(&c[0], private_sum);
    }
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


/*inline void atomicAdd_g_d(volatile __global double *addr, double val)
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
}*/

void atomicAdd_g_d(__global double *val, double delta) {
  union {
    double f;
    ulong  i;
    } old;
  union {
    double f;
    ulong  i;
    } new;
  do {
    old.f = *val;
    new.f = old.f + delta;
   } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
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



__kernel void double_dot_matrix(const int M, const int K, const int N, const int TS,
                      __global double* A,
                      __global double* B,
                      __global double* C,
                      __local double* Asub,
                      __local double* Bsub) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Initialise the accumulation register
    double acc = 0;
    
    // Loop over all tiles
    const int numTiles = ceil((float) K/TS);
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        Asub[col + row*TS] = A[globalRow*K + t*TS + col];
        Bsub[col + row*TS] = B[N*(t*TS + row) + globalCol];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        const bool change_size = (t+1)/numTiles;
        int new_size = K - (numTiles - 1)*TS;

        //local size = new_size if change_size == True, local_size otherwise
        new_size = change_size * new_size + (1 - change_size) * TS;

        // Perform the computation for a single tile
        for (int k=0; k<new_size; k++) {
            acc = fma(Asub[k + row*TS], Bsub[col + k*TS], acc);
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    if(globalCol < N && globalRow < M)
      C[globalCol + M*globalRow] = acc;
}


__kernel void double_matrix_vec( __global double* A, __global double* vec, __global double* c,
			  __local double* Asub, __local double* vecsub,
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
	  double private_sum = 0.0f;
	  const int num_groups = get_num_groups(1) - 1;
	  const int group_id = get_group_id(1);
	  const bool change_size = group_id/num_groups;
	  int new_size = vec_size - num_groups*local_size_cols;

	  //local size = new_size if change_size == True, local_size otherwise
	  new_size = change_size * new_size + (1 - change_size) * local_size_cols;
	  for(int i = 0; i < new_size; i++) {
	    private_sum += Asub[row*local_size_cols + i];
	  }
	  atomicAdd_g_d(&c[globalRow], private_sum);
	}
      }
    }
  }
}

__kernel void double_vec_dot(__global double* a, __global double* b, __global double* c,
		      __local double* local_sum, int vec_size) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);
  const int local_size = get_local_size(0);

  if(gid < vec_size) {
    local_sum[lid] = a[gid] * b[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0) {
      double private_sum = 0.0f;

      const int num_groups = get_num_groups(0) - 1;
      const int group_id = get_group_id(0);
      const bool change_size = group_id/num_groups;
      int new_size = vec_size - num_groups*local_size;

      //local size = new_size if change_size == True, local_size otherwise
      new_size = change_size * new_size + (1 - change_size) * local_size;
      
      for(int i = 0; i < new_size; i++) {
	private_sum += local_sum[i];
      }
      atomicAdd_g_d(&c[0], private_sum);
    }
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
