#include "name_func.h"

#define FUNCNAME NAME1(negative, Type1)

__kernel void FUNCNAME(__global Type1 *a_n, const __global Type1 *a) {
  int gid = get_global_id(0);
  a_n[gid] = -a[gid];  
}

#undef FUNCNAME