    __global__ void Suma(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }
    
    __global__ void ISuma(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }

    __global__ void Resta(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    } 
    
    __global__ void IResta(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    }
    

    __global__ void Multi(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }
    
    __global__ void IMulti(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }

    __global__ void Divide(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }
    
    __global__ void IDivide(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }

    __global__ void Cross(int sizean, int sizeam, int sizebm,float *a, float *b, float *c)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < sizebm && ty < sizean){
            float Pvalue = 0;
            for (int k = 0; k < sizeam; ++k) {
                float Aelement = a[ty * sizeam + k];
                float Belement = b[k * sizebm + tx];
                Pvalue += Aelement * Belement;
            }
            c[ty * sizebm + tx] = Pvalue;
        }
    }

    // dominante modificando la matriz original
    __global__ void Dom1(int size, float *a)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        float Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
            }
        a[ty * size + ty] = Pvalue + 2000.0;
        }
    }

    // dominante guardando el resultado en otra matriz
    __global__ void Dom2(int size, float *a, float *b)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        float Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
                b[ty * size + i] = a[ty * size + i];
            }
            b[ty * size + ty] = Pvalue + 2000.0; 
        }
    }
    
    __global__ void Transpose(int size, const float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[ty + (tx*size)] =  a[tx + (ty*size)];
        }
    }
    
    __global__ void neg(int size, float * a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  a[tx + (ty*size)] * -1;
        }
    }
    
    __global__ void absolute(int size, float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  abs(a[tx + (ty*size)]);
        }
    }

    // metodos para diagonales 

    __global__ void Diag(int size, float *a, float *b)
    {
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idy < size){
            b[idy] =  a[idy + (idy*size)];
        }
    }

    __global__ void DiagFlat(int size, float *a, float *b)
    {
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idy < size){
            b[idy + (idy*size)] =  a[idy];
        }
    }