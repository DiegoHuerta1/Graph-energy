import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json



from funciones_grafos import *


'''
Creacion de conjunto del json de grafos con estadisticas
'''

# definir donde se guarda el output
carpeta_datos = "./datos/"
output_filename = "datos_grafos_json"


# tomar los grafos
lista_grafos = obtener_grafos_conexos_pequeños()


# ver cuantos son
numero_grafos = len(lista_grafos)
print(f"Se crean estadisticas de {numero_grafos} grafos\n")


# hacer una lista con las estadisticas de todos los grafos
lista_estadisticas = []


# iterar en los grafos
for idx_g, grafo in enumerate(lista_grafos):
    
    # obtener las estadisticas
    estadisticas_grafo = (obtener_estadisticas_grafo(grafo))
    
    # hacer que pueda ser serializable con json
    
    # transformar los arrays en listas
    # transformar np.bool_ en bool
    
    # pasarlo a otro dict con buen formato
    estadisticas_grafo_formato = {}
    
    for key, value in estadisticas_grafo.items():
        
        # si es array
        if isinstance(value, np.ndarray):
            # hacerlo lista
            estadisticas_grafo_formato[key] = value.tolist()
            
        # si es np.bool_ en bool
        elif isinstance(value, np.bool_):
            estadisticas_grafo_formato[key] = bool(value)
            
        # dejar los demas igual
        else:
            estadisticas_grafo_formato[key] = value

    
    # agregar las estadisticas de este grafo
    lista_estadisticas.append(estadisticas_grafo_formato)
    
    # marcar progreso
    print(f"Progreso: [{idx_g+1}/{numero_grafos}]")
    
    
    
# guardar la lista de diccionarios
with open(carpeta_datos + output_filename, 'w') as fout:
    json.dump(lista_estadisticas, fout)
    
    
    
    
    
    
    
    
    
    
    