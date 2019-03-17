#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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