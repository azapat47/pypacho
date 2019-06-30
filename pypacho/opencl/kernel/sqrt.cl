#include "name_func.h"

#define FUNCNAME NAME1(sqrt, Type1)

__kernel void FUNCNAME(__global Out_Type *a_n, const __global Type1 *a) {
  int gid = get_global_id(0);
  a_n[gid] = sqrt((Out_Type) a[gid]);    
}

#undef FUNCNAME