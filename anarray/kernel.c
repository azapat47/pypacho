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

__kernel void divide(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] / b[gid];
}

__kernel void dot(__global float *a,__global float *b, __global float *c,
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

/*__kernel void mod(__global float *a, __global float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] % b[gid];
  }*/

__kernel void negative(__global float *a_n, __global float *a, unsigned a_width, unsigned a_height)
{
  int gid = get_global_id(0);
  a_n[gid] = -a[gid];             
}
