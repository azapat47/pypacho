#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#include "atomics.cl"

#define Type1 float
#define Type2 float
#define Out_Type float

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"
#include "transpose.cl"
#include "negative.cl"
#include "diag.cl"
#include "diagflat.cl"
#include "sqrt.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 float
#define Type2 double
#define Out_Type double

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 float
#define Type2 int
#define Out_Type float

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 double
#define Type2 float
#define Out_Type double

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"
#include "transpose.cl"
#include "negative.cl"
#include "diag.cl"
#include "diagflat.cl"
#include "sqrt.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 double
#define Type2 double
#define Out_Type double

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 double
#define Type2 int
#define Out_Type double

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/


#define Type1 int
#define Type2 float
#define Out_Type float

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"
#include "transpose.cl"
#include "negative.cl"
#include "diag.cl"
#include "diagflat.cl"
#include "sqrt.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 int
#define Type2 double
#define Out_Type double

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 

/*--------------------------------------*/

#define Type1 int
#define Type2 int
#define Out_Type int

#include "add.cl"
#include "subtract.cl"
#include "multiply.cl"
#include "divide.cl"
#include "scalar_mult.cl"
#include "dot_matrix.cl"
#include "vec_dot.cl"
#include "matrix_vec.cl"

#undef Type1 
#undef Type2 
#undef Out_Type 
