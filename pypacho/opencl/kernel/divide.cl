#include "name_func.h"

#define FUNCNAME NAME(divide, Type1, Type2) 

__kernel void FUNCNAME(const __global Type1 *a, const __global Type2 *b, __global Out_Type *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] / b[gid];
}

#undef FUNCNAME