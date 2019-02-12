# hairyNumbers
Análisis Numérico - Métodos Numéricos implementados en OpenCL y CUDA

## Contributors

Andrés Felipe Zapata Palacio
Esteban Echeverri Jaramillo
Diego Alejandro Cifuentes
Juan David Arcila Moreno

## USAGE

Esta guia asume la instalación correcta de `Cuda toolkit` en cualquier version >= 2.0

### Verficiar la instalación correcta de cuda

    nvcc -V
    
### Install pycuda

Recomendamos el uso de `conda` para la instalación. Python >= 3.

     conda create -n pypacho numpy boost pandas mathplotlib

     conda activate pypacho

Ahora descargamos el respositorio oficial de pycuda e ingresamos

     git clone http://git.tiker.net/trees/pycuda.git
     cd pycuda

Despues configuramos la compilación. --cuda-root debe ser la carpeta de instalación de cuda que contiene las demás carpetas como 'lib64', 'bin', etc.
--cudadrv-lib-dir debe apuntar al directirio que incluye todo los .so del driver
--boots-python-libname y  --boost-thread-libname debería funcionar sin ningun cambio usando el ambiente de conda

     ./configure.py --cuda-root=<path_to_cuda> --cudadrv-lib-dir=/usr/lib64/nvidia --boost-python-libname=boost_python-mt --boost-thread-libname=boost_thread-mt
     make

Es buena idea verificar que el fichero setup.cfg y verficar los valores generados

Despues compilamos e instalamos. La instalación, de usar el ambiente de conda, quedará dentro de los paquetes del ambiente creado (`pypacho`)

     make install

por último corremos las pruebas para garantiza la integridad de la compilación

     make tests

### Correr laboratorios

Casos

labopratorio metodos numericos

cuda y numpy

python3 laboratory.py "[intentos por tamaño, tamaño inicial, delta, cuantos tamaños van a ser probados, iteraciones maximas por metodo, tolerancia de los metodos],[cuda, opencl,numpy],[jacoi,gradiente decendiente, gradiente conjugado]"

cuda, opencl, numpy

optirun python3 laboratory.py "[intentos por tamaño, tamaño inicial, delta, cuantos tamaños van a ser probados, iteraciones maximas por metodo, tolerancia de los metodos],[cuda, opencl,numpy],[jacoi,gradiente decendiente, gradiente conjugado]"

ejemplo

laboratorio que probara los metodos jacobi y gradiente conjugado en cuda y opencl, empezando con una matriz de tamaño 10 y amuentando de 10 en 10 hasta 50 y por cada tamaño se haran 5 corridas

optirun python3 laboratory.py "[5, 10, 10, 5, 100,0.000001],[1,1,0],[1,0,1]"

laboratorio de operaciones

cuda y numpy

python3 operationslab.py "[1,10,10,10,10,10,5]" "[1,0,0]" "[1,0,0,0,0,0,0]"

cuda, opencl y numpy

optirun python3 operationslab.py "[1,10,10,10,10,10,5]" "[1,0,0]" "[1,0,0,0,0,0,0]"
