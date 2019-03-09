    __global__ void Suma(int t_a, int t_b, int size_n, int size_m, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(idx < size_m && idy < size_n){
            if(t_a == 0){
                ida = idx + (idy*size_m); 
            }
            else{
                ida = idy + (idx*size_n);
            }
            if(t_b == 0){
                idb = idx + (idy*size_m);
            }
            else{
                idb = idy + (idx*size_n); 
            }
            c[idx + (idy*size_m)] =  a[ida] + b[idb];
        }
    }

    __global__ void Resta(int t_a, int t_b, int size_n, int size_m, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(idx < size_m && idy < size_n){
            if(t_a == 0){
                ida = idx + (idy*size_m); 
            }
            else{
                ida = idy + (idx*size_n);
            }
            if(t_b == 0){
                idb = idx + (idy*size_m);
            }
            else{
                idb = idy + (idx*size_n); 
            }
            c[idx + (idy*size_m)] =  a[ida] - b[idb];
        }
    }   

    __global__ void Multi(int t_a, int t_b, int size_n, int size_m, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(idx < size_m && idy < size_n){
            if(t_a == 0){
                ida = idx + (idy*size_m); 
            }
            else{
                ida = idy + (idx*size_n);
            }
            if(t_b == 0){
                idb = idx + (idy*size_m);
            }
            else{
                idb = idy + (idx*size_n); 
            }
            c[idx + (idy*size_m)] =  a[ida] * b[idb];
        }
    }

    __global__ void RMulti(int t_a, int t_b, int size_n, int size_m, double b, double *a, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size_m && idy < size_n){
            c[idx + (idy*size_m)] =  a[idx + (idy*size_m)] * b;
        }
    }

    __global__ void Divide(int t_a, int t_b, int size_n, int size_m, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(idx < size_m && idy < size_n){
            if(t_a == 0){
                ida = idx + (idy*size_m); 
            }
            else{
                ida = idy + (idx*size_n);
            }
            if(t_b == 0){
                idb = idx + (idy*size_m);
            }
            else{
                idb = idy + (idx*size_n); 
            }
            c[idx + (idy*size_m)] =  a[ida] / b[idb];
        }
    }

    __global__ void DOT(int t_a, int t_b, int sizean, int sizeam, int sizebm,double *a, double *b, double *c)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(tx < sizebm && ty < sizean){
            double Pvalue = 0;
            for (int k = 0; k < sizeam; ++k) {
                if(t_a == 0){
                    ida = ty * sizeam + k; 
                }
                else{
                    ida = k * sizean + ty;
                }
                if(t_b == 0){
                    idb = k * sizebm + tx;
                }
                else{
                    idb = tx * sizeam + k; 
                }
                double Aelement = a[ida];
                double Belement = b[idb];
                Pvalue += Aelement * Belement;
            }
            c[ty * sizebm + tx] = Pvalue;
        }
    }

    // dominante modificando la matriz original
    __global__ void Dom1(int size, double *a)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        double Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
            }
        a[ty * size + ty] = Pvalue + 2000.0;
        }
    }

    // dominante guardando el resultado en otra matriz
    __global__ void Dom2(int size, double *a, double *b)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        double Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
                b[ty * size + i] = a[ty * size + i];
            }
            b[ty * size + ty] = Pvalue + 2000.0; 
        }
    }
    
    __global__ void Transpose(int sizem, int sizen, const double *a, double *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < sizem && ty < sizen){
            b[ty + (tx*sizen)] =  a[tx + (ty*sizem)];
        }
    }

    __global__ void neg(int size_n, int size_m, double * a, double *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size_m && ty < size_n){
            b[tx + (ty*size_m)] =  a[tx + (ty*size_m)] * -1;
        }
    }
    
    __global__ void absolute(int size_n, int size_m, double *a, double *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size_m && ty < size_n){
            b[tx + (ty*size_m)] =  abs(a[tx + (ty*size_m)]);
        }
    }

    // metodos para diagonales 

    __global__ void Diag(int size, double *a, double *b)
    {
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idy < size){
            b[idy] =  a[idy + (idy*size)];
        }
    }

    __global__ void DiagFlat(int size, double *a, double *b)
    {
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idy < size){
            b[idy + (idy*size)] =  a[idy];
        }
    }

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMul(double * A, double * B, double * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns,
                   int t_a, int t_b) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    double Pvalue = 0;
    int ida = 0;
    int idb = 0;
    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns){
           if(t_a == 0){
                ida = Row*numAColumns + (m*TILE_WIDTH+tx);
            }
                else{
                ida = (m*TILE_WIDTH+tx)*numARows + Row;
            }
          ds_M[ty][tx] = A[ida];
       }
       else{
          ds_M[ty][tx] = 0;
       }
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows){
           if(t_b == 0){
                idb = (m*TILE_WIDTH+ty)*numBColumns+Col;
            }
                else{
                idb = Col*numBRows+(m*TILE_WIDTH+ty);
            }
          ds_N[ty][tx] = B[idb];
       }
       else{
          ds_N[ty][tx] = 0;
       }

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}


// atomicadd sacado de https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
// vec_dot sacado de https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
// matrixMul sacado de https://gist.github.com/wh5a/4313739

// Compute C = A * B


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }


  #endif

#define N 32

__global__ void vec_dot(double * a, double * b, double * c, int size,
                   int t_a, int t_b) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ double temp[N];
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    temp[threadIdx.x] = a[tx] * b[tx];

    __syncthreads();


    if(threadIdx.x == 0){
        double sum = 0;
        for(int i = 0; i < N; i++){
            if(tx + i < size){
                sum += temp[i];
            }
        }
        atomicAdd(c,sum);
    }
}

// matrix * vector

__global__ void MatDotVec(double * A, double * B, double * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns,
                   int t_a, int t_b) {
    __shared__ double sub_mat[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sub_vec[TILE_WIDTH];

    int by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty;

    double Pvalue = 0;
    int ida = 0;
    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns){
           if(t_a == 0){
                ida = Row*numAColumns + (m*TILE_WIDTH+tx);
            }
                else{
                ida = (m*TILE_WIDTH+tx)*numARows + Row;
            }
           sub_mat[ty][tx] = A[ida];
       }
       else{
          sub_mat[ty][tx] = 0;
       }
       if(m*TILE_WIDTH+ty<numBRows){
           sub_vec[ty] = B[m*TILE_WIDTH+ty];
       }
       else{
           sub_vec[ty]= 0; 
       }

       __syncthreads();
       
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += sub_mat[ty][k] * sub_vec[k];
       __syncthreads();
    }
    if (Row < numCRows)
       C[Row] = Pvalue;
}