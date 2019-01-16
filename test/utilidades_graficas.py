import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def graficar(tabla, metodo='jacobi', x='tamaño', y='tiempo_promedio', plataformas=['cuda', 'opencl', 'numpy']):
    cmap = plt.cm.get_cmap('hsv', len(plataformas) + 1)
    
    for indice, plataforma in enumerate(plataformas):
        tabla_plataforma = tabla.loc[tabla['plataforma'] == plataforma]
        tabla_metodo = tabla_plataforma.loc[tabla_plataforma['metodo'] == metodo]
        x_plataforma = tabla_metodo[x].values
        y_plataforma = tabla_metodo[y].values
        plt.plot(x_plataforma, y_plataforma, label=plataforma, color=cmap(indice))
    plt.title(y + ' vs ' + x + ' en ' + metodo)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.tight_layout()
    plt.show()

def describir(tabla):
    promedio = tabla.mean()
    std = tabla.std()
    return [promedio, std]