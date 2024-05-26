import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from funciones_grafos import *


'''
Creacion de conjunto de datos de grafos con estadisticas
'''

# definir donde se guarda el output
carpeta_datos = "./datos/"
output_filename = "datos_grafos.csv"



# tomar los grafos
lista_grafos = obtener_grafos_conexos_pequeños()


# ver cuantos son
numero_grafos = len(lista_grafos)
print(f"Se crean estadisticas de {numero_grafos} grafos\n")


# hacer una lista con las estadisticas de todos los grafos
lista_estadisticas = []


# iterar en los grafos
for idx_g, grafo in enumerate(lista_grafos):
    
    # agregar las estadisticas de este grafo
    lista_estadisticas.append(obtener_estadisticas_grafo(grafo))
    
    # marcar progreso
    print(f"Progreso: [{idx_g+1}/{numero_grafos}]")
    
    
    
    
# ya que se tien la lista de diccionarios, hacerla df
df_grafos = pd.DataFrame(lista_estadisticas)


# guardar el df
df_grafos.to_csv(carpeta_datos+output_filename)


