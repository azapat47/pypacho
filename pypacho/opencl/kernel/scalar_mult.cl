#include "name_func.h"

#define FUNCNAME NAME(scalar_mult, Type1, Type2)

__kernel void FUNCNAME (const __global Type1 *a, Type2 b, volatile __global Out_Type *c) 
{
  int gid = get_global_id(0);
  c[gid] = a[gid] * b;
}

#undef FUNCNAME