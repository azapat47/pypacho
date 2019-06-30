#include "name_func.h"

#define FUNCNAME NAME1(transpose, Type1)

__kernel void FUNCNAME(__global Type1 *a_t, const __global Type1 *a,
                       unsigned a_width, unsigned a_height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx_a = row + col*a_height;
  int idx_a_t = col + row*a_width;
  a_t[idx_a_t] = a[idx_a];
}

#undef FUNCNAME