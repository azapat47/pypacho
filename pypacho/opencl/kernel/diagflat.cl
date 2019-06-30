#include "name_func.h"

#define FUNCNAME NAME1(diagflat, Type1)

__kernel void FUNCNAME(const __global Type1 *a, __global Type1 *b, int a_size) {
  int gid = get_global_id(0);
  if(gid < a_size){
    b[gid*a_size + gid] = a[gid];
  }
}

#undef FUNCNAME