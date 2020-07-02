# PyPacho
Análisis Numérico - Métodos Numéricos implementados en OpenCL y CUDA

## Autores

Andrés Felipe Zapata Palacio
Esteban Echeverri Jaramillo
Diego Alejandro Cifuentes
Juan David Arcila Moreno

## Uso

Esta guia asume la instalación correcta de `Cuda Toolkit` en cualquier version >= 2.0

### Verficiar la instalación correcta de Cuda

    nvcc -V

#### Creación del Ambiente virtual de Python

Recomendamos el uso de `conda` para la instalación. Se necesita una versión `Python>=3`.

     conda create -n pypacho python=3.7 

     conda activate pypacho


### Instalación de Pre-requisitos
 
Las últimas versiones de pycuda y pyopencl permiten su instalación por medio de ´pip´ haciendo uso los repositorios oficiales de Python. Ya no es necesario hacer la compilación de ninguno de los paquetes. 

    pip install pycuda pyopencl

Adicionalmente, para poder recuperar los datos de los laboratorios se hace uso de la libreria `pandas`.

     pip install pandas

### Ejecución de los Laboratorios

Los laboratorios constan de tres partes: 

1. Laboratorio de limites: Permite conocer el tamaño máximo del sistema de ecuaciones que se puede ejecutar.

Modo de ejecucion:

     python limits_lab.py

En la salida estandar se verá el progreso del laboratorio y al final se imprimirá un diccionario dando los limites encontrados por cada método en cada plataforma.

2. Laboratorio de Métodos: Permite saber el tiempo de ejecución, número de iteraciones, disperción y error en métodos numericos para solucionar sistemas de ecuaciones lineales. Se evaluan diferentes plataformas con diferentes tamaños y numero de intentos.

**Modo de ejecución:**

    python laboratory.py "[intentos por tamaño, tamaño inicial, delta, cuantos tamaños van a ser probados, iteraciones maximas por metodo, tolerancia de los metodos]" "[cuda, opencl,numpy]" "[jacoi,gradiente decendiente, gradiente conjugado]" "Flag precision doble o simple"


**Ejemplo**  

Este laboratorio probará los métodos jacobi y gradiente conjugado en Cuda y OpenCL, empezando con una matriz de tamaño 10 y aumentando de 10 en 10 hasta 50. Por cada tamaño se harán 6 corridas distintas. Usará presición doble. 

	python laboratory.py "[6, 10, 10, 5, 100,0.000001]" "[1,1,0]" " [1,0,1]" "true"

3. Laboratorio de Operaciones: Permite saber el tiempo de ejecución de las operaciones básicas en las distintas plataformas a evaluar. 

**Modo de ejecución**

    python operationslab.py "[intentos por tamaño, tamaño inicial filas matriz a, tamaño columnas matriz a, tamaño inicial filas matriz b, tamaño inicial columnas matriz b,  delta,  cuantos tamaños van a ser probados]"  "[cuda, opencl, numpy]" "[ +, -, @, /,  *, transp, norm]" "flag precision doble o simple"
**Ejemplo**

Este laboratorio probará la operación de suma y norma, empezando con matrices de tamaño [10,10] y [10,10] en la plataforma Cuda y Numpy. Se probarán 5 tamaños, aumentando de 10 en 10 hasta 50. Se harán 3 corridas distintas por cada tamaño. Usará presicion simple.

	python operationslab.py "[3,10,10,10,10,10,5]" "[1,0,1]" "[1,0,0,0,0,0,1]" "false"


