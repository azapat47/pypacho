    __global__ void Suma(int size, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }
    
    __global__ void ISuma(int size, double *a, double *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }

    __global__ void Resta(int size, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    } 
    
    __global__ void IResta(int size, double *a, double *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    }
    

    __global__ void Multi(int size, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }

    __global__ void RMulti(int size,  double b, double *a, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] * b;
        }
    }
    
    __global__ void IMulti(int size, double *a, double *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }

    __global__ void Divide(int size, double *a, double *b, double *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }
    
    __global__ void IDivide(int size, double *a, double *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }

    __global__ void Cross(int sizean, int sizeam, int sizebm,double *a, double *b, double *c)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < sizebm && ty < sizean){
            double Pvalue = 0;
            for (int k = 0; k < sizeam; ++k) {
                double Aelement = a[ty * sizeam + k];
                double Belement = b[k * sizebm + tx];
                Pvalue += Aelement * Belement;
            }
            c[ty * sizebm + tx] = Pvalue;
        }
    }

    __global__ void multiply(int n, int m, int p, double *a, double *b, double *c)
    {
        int idx = p*threadIdx.x + threadIdx.y;

        c[idx] = 0.0;
        for(int k=0; k<m; k++)
           c[idx] += a[m*threadIdx.x+k]
                    *b[threadIdx.y+k*p];
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
    
    __global__ void neg(int size, double * a, double *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  a[tx + (ty*size)] * -1;
        }
    }
    
    __global__ void absolute(int size, double *a, double *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  abs(a[tx + (ty*size)]);
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