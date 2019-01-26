    __global__ void Suma(int t_a, int t_b, int size_n, int size_m, float *a, float *b, float *c)
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

    __global__ void Resta(int t_a, int t_b, int size_n, int size_m, float *a, float *b, float *c)
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

    __global__ void Multi(int t_a, int t_b, int size_n, int size_m, float *a, float *b, float *c)
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

    __global__ void RMulti(int t_a, int t_b, int size_n, int size_m, float b, float *a, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size_m && idy < size_n){
            c[idx + (idy*size_m)] =  a[idx + (idy*size_m)] * b;
        }
    }

    __global__ void Divide(int t_a, int t_b, int size_n, int size_m, float *a, float *b, float *c)
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

    __global__ void Cross(int t_a, int t_b, int sizean, int sizeam, int sizebm,float *a, float *b, float *c)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        int ida = 0;
        int idb = 0;
        if(tx < sizebm && ty < sizean){
            float Pvalue = 0;
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
                float Aelement = a[ida];
                float Belement = b[idb];
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
    
    __global__ void Transpose(int sizem, int sizen, const float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < sizem && ty < sizen){
            b[ty + (tx*sizen)] =  a[tx + (ty*sizem)];
        }
    }

    __global__ void neg(int size_n, int size_m, float * a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size_m && ty < size_n){
            b[tx + (ty*size_m)] =  a[tx + (ty*size_m)] * -1;
        }
    }
    
    __global__ void absolute(int size_n, int size_m, float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size_m && ty < size_n){
            b[tx + (ty*size_m)] =  abs(a[tx + (ty*size_m)]);
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