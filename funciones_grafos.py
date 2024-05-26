# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:34:34 2024

@author: diego
"""


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
from scipy.optimize import minimize
import math
from networkx.algorithms import bipartite
import scipy



'''
Funciones auxiliares analisis cota energia grafos
'''

def vector_to_matrix(vector):
    length = len(vector)
    size = int(np.sqrt(length))
    rows = size
    cols = size
    
    return np.array(vector).reshape(rows, cols)




def matrix_to_vector(matrix):
    return matrix.flatten()




# calcular el indice de randic de un grafo
def get_randic(grafo):
    
    # comienza en 0
    randic = 0
    
    # iterar en las aristas
    for edge in grafo.edges():
        
        # tomar los vertices de la arista
        node1, node2 = edge
        
        # tomar sus grados
        degree_node1 = grafo.degree(node1)
        degree_node2 = grafo.degree(node2)
        
        
        # sumar el termino de la arista
        randic = randic + (1/np.sqrt(degree_node1*degree_node2))
        
    return randic




# funcion para dibujar el grafo
def dibujar_grafo(grafo,  fig_size=(6, 6), pos = None):
    
    plt.figure(figsize=fig_size)
    
    if pos is None:
        # Dibujar nodos y aristas
        nx.draw(grafo, pos=nx.kamada_kawai_layout(grafo), with_labels=True, node_size=700, 
                node_color='skyblue')
        
    if pos is not None:
        nx.draw(grafo, pos=pos, with_labels=True, node_size=700, 
                node_color='skyblue')
        

    # Mostrar el grafo
    plt.show()
    
    
    
    
def create_dandelion_graph(n):
    
    # crear vacio
    G = nx.Graph()
    
    # calcular nuemro de vertices
    num_vertices = 3*n+1
    
    # ponerlos
    G.add_nodes_from(list(range(1, num_vertices+1)))
    
    
    # hacer las conexiones
    for i in range(2, n+2):
        
        # conectar el nodo central a los petalos
        G.add_edge(1, i)
        
        # conectar los petalos
        G.add_edge(i, n+i)
        G.add_edge(i, 2*n+i)
        
    return G
    

def get_energia_grafo(grafo):
    
    # obtener la matriz de adjacencia
    adj  = nx.to_numpy_array(grafo)
    
    eigenval = np.linalg.eigvalsh(adj)
    
    # regresar la energia
    return sum(abs(eigenval))



def analisis_pesos_cota(G, repeticiones = 10, ver_desarrollo=0, devolver = 0):
    
    '''
    Argumentos:
    G - grafo
    repeticiones - numero de veces que se intenta la optimizacion
    devolver - determinar si se necesita devolver algo
    ver_desarrollo - variable boleana para ver el proceso mas detallado
                    -1 : no se muestra nada (por si solo se quiere devoler)
                     0 : se muestra el grafo original, el grafo de mejor randic,
                         el grafo de mejor cota, se imprimen cantidades
                     1 : informacion de 0, y ademas se muestran las
                         las repeticiones de la optimizacion
                     2 : informacion de 1, y ademas se muestra
                         el indice de randic de todos los subgrados
    '''
    
    
    # ver que se tena un valor correcto
    assert ver_desarrollo in [-1, 0, 1, 2]

    # tomar las aritas
    edges_grafo = list(G.edges)
    
    
    # hacer el pos del grafo para dibujar todo
    pos = nx.kamada_kawai_layout(G)

    # ver todas las posibles combinaciones
    combinaciones_aristas = list(chain.from_iterable(combinations(edges_grafo, r) 
                                                for r in range(1, len(edges_grafo)+1)))
    # ver
    if ver_desarrollo >= 2:
        print(f"El grafo tiene {len(edges_grafo)} aristas")
        print(edges_grafo)
        print(" ")
        print(f"Por lo tanto, hay {len(combinaciones_aristas)} combinaciones de aristas")


    # guardar el grafo con el mayor indice de randic
    # y el indice de randic mayor
    mayor_indice_randic = 0
    grafo_mayor_randic = None

    # iterar en la combinacion de las aristas
    for aristas in combinaciones_aristas:

        # hacerlo lista
        aristas = list(aristas)

        # decir
        if ver_desarrollo >= 2:
            print("-"*100)
            print("Considerar las aristas: ")
            print(aristas)
            print(" ")

            # considerar un grafo solo con esas aristas
            print("Constuir un grafo con solo esas aristas")

        # constuir el grafo
        grafo_aristas = nx.Graph()
        
        # poner los nodos del grfo original
        grafo_aristas.add_nodes_from(G.nodes)
        
        # poner las aristas
        grafo_aristas.add_edges_from(aristas)

        # ver el grafo
        if ver_desarrollo >= 2:
            dibujar_grafo(grafo_aristas, (2, 2), pos=pos)

        # calcular el indice de randic
        randic_grafo_aristas = get_randic(grafo_aristas)

        #imprimir
        if ver_desarrollo >= 2:
            print(f"El indice de randic de este grafo es: {randic_grafo_aristas}")

        # ver si es el mayor
        if randic_grafo_aristas > mayor_indice_randic:
            # ponerlo como mayor
            mayor_indice_randic = randic_grafo_aristas
            # guardar le grafo
            grafo_mayor_randic = grafo_aristas


    if ver_desarrollo >= 2:     
        print("-"*200)

        
    # calcular la mejor cota con randic
    mejor_cota_randic = 2*mayor_indice_randic
        
    # ver el original con su randic
    original_randic = get_randic(G)
    
    if ver_desarrollo >= 0:
        print(f"El grafo original tiene indice de randic: {original_randic}")
        dibujar_grafo(G, (4, 4), pos=pos)

        # ver que significa en terminos de la cota
        print(f"Es decir, la cota del indice de randic es: {2*original_randic}")

        print("-"*50)

        # ver donde se alcanzo el mayor
        print(f"El mayor indice de randic es: {mayor_indice_randic}")
        print("Se alcanza con el grafo")
        dibujar_grafo(grafo_mayor_randic, (4, 4), pos=pos)

        # ver que significa en terminos de la cota
        print(f"Es decir, la cota del indice de randic es: {mejor_cota_randic}")

    #--------------------------------------------------------------------------------------

    # los pesos se representan como matriz
    # la entrada i,j es el peso del nodo i al nodo j

    
    # problema de optimizacion
    if ver_desarrollo >= 0:
        print("-"*500)
        print("Optimizar los pesos")

    # sacar la matriz de adj del grafo
    adj_matrix  = nx.to_numpy_array(G)

    
    # ver cuantos nodos hay
    num_nodos = G.number_of_nodes()
    

    
    # ver la funcion objetivo
    # dados unos pesos ver la cota
    def objective_function(pesos_vector):

        # hacer que los pesos sean matriz
        pesos_matriz = vector_to_matrix(pesos_vector)

        # por cada par de nodos conectado
        # multiplicar el peso de uno por el de otro
        pesos_por_edge = adj_matrix*pesos_matriz*pesos_matriz.T
        
        # si hay un valor menor a 0, devolver un valor grande
        if np.any(pesos_por_edge < 0):
            return float('inf')  # Devuelve infinito si hay algún elemento negativo

        # se saca raiz a cada uno y se suma
        cota = np.sqrt(pesos_por_edge).sum()

        # se esta considerando cada edge dos veces
        # entonces no se debe multiplicar por 2
        # poner un menos para hacer minimizacion
        return -cota
    

    # poner las restricciones

    # todas las entradas deben ser mayores a 0
    def positive_entries(pesos_vector):
        return pesos_vector


    # solo debe haber pesos a nodos conetados
    def restriccion_local(pesos_vector):

        # tomar la matriz de los pesos
        pesos_matriz = vector_to_matrix(pesos_vector)

        # tomar el complemento de la matriz de adyacenica
        complemento_adj = 1 - adj_matrix

        # no deberia de haber pesos en estas entradas
        # ver los pesos que hay
        pesos_incorrectos = complemento_adj * pesos_matriz

        # sumar todos los incorrectos
        suma_pesos_incorrectos = sum(pesos_incorrectos)

        # esta suma deberia ser igual a 0 (no pesos incorrectos)
        return suma_pesos_incorrectos


    # la suma de las filas debe ser 1
    def suma_filas(pesos_vector):
        # hacerlo matriz
        pesos_matriz = vector_to_matrix(pesos_vector)

        return np.sum(pesos_matriz, axis=1) - 1

    # Definir restricciones
    constraints = [{'type': 'eq', 'fun': suma_filas},
                   {'type': 'eq', 'fun': restriccion_local},
                   {'type': 'ineq', 'fun': positive_entries}]

    
    
    # opciones para la minimizacion
    options = {'ftol': 1e-10, 'maxiter': 10000}
    
        
    # hacer la optimizacion varias veces
    # reportar el mejor resultado
    
    # ir guardande el mejor resultado
    best_result = None
    # ir guardando la mejor cota hasta ahora
    best_cota_repeticiones = float('-inf')
    
    
    # hacer las repeticiones de la optimizacion
    for idx_repeticion in range(repeticiones):
        
        if ver_desarrollo >= 1:
            print(f"Intento {idx_repeticion} de la optimizacion.")
            
        # si es el primero intento
        # poner x_0 a la matriz de adj
        # despues que sea aleatorio
        if idx_repeticion == 0:
            pesos_iniciales_vector = matrix_to_vector(adj_matrix)
        else:
            pesos_iniciales_vector = matrix_to_vector(np.random.rand(num_nodos, num_nodos))
        
           

        # realizar la maximizacion (minimizacion por poner el - en la funcion objetivo)
        result_repeticion = minimize(objective_function, pesos_iniciales_vector, 
                          constraints=constraints, options=options)


        # tomar el valor optimizado de la cota
        cota_optimizada_repeticion = -result_repeticion.fun
        
        
        # reportarlo
        if ver_desarrollo >= 1:
            print(f"Cota alcanzada: {round(cota_optimizada_repeticion, 5)}")
            print("-"*20)
        
        
        # ver si es el mejor
        if cota_optimizada_repeticion > best_cota_repeticiones:
            
            # actualizar el mejor resutlado y mejor cota
            best_result = result_repeticion
            best_cota_repeticiones = cota_optimizada_repeticion
    
    
    
    # poner una linea separadora
    if ver_desarrollo >= 1:
        print("-"*50)
        
    
    # el resultado de la optimizacion es el mejor resultado de las repeticioens
    # ver que se alcanzo
    cota_optimizada = round(-best_result.fun, 8)
    
    
    # reportarlo
    if ver_desarrollo >= 0:
        print(f"Valor de la cota con los pesos optimos: {cota_optimizada}")
    
    
    # ver la solucion optima
    optimal_solution = best_result.x

    # redondear los pesos y hacerlo matriz
    optimal_solution = vector_to_matrix(np.round(optimal_solution, 3))

    # ver los pesos que hacen optimo
    if ver_desarrollo >= 0:
        print("")
        print("Pesos optimos:")
        print(optimal_solution)
        print("")
    

        # con los pesos optimos constuir un grafo
        grafo_opti = nx.DiGraph()

        # los nodos de este grafo inician con el mismo indice
        # que el grafo original, es decir, indice 0 o indice 1
        # este es el offset
        # indice 0 - offset = 0
        # indice 1 - offset = 1
        # se calcula viendo el nodo menor
        offset = min(G.nodes)


        # añadir los nodos del grafo, indice 1
        grafo_opti.add_nodes_from(range(offset, num_nodos + offset))

        # iterar en los pesos optimos, ir construyendo el grafo
        for i in range(num_nodos):
            for j in range(num_nodos):
                # tomar el peso de i a j
                weight = optimal_solution[i][j]

                # ver si es mayor a 0 para ponerlo
                if weight > 0:

                    # añadir la arista con el peso al grafo
                    # ajustar indices si es necesario
                    grafo_opti.add_edge(i + offset, j + offset, weight=weight)


        # ver el grafo optimo creado
        plt.figure(figsize=(4, 4))
        nx.draw(grafo_opti, pos=pos, with_labels=True, node_size=1000, edge_color='black', node_color='skyblue',
                width=[edge[2]['weight'] * 5 for edge in grafo_opti.edges(data=True)],
                arrowsize=20, connectionstyle='arc3,rad=0.1')
        plt.show()
    
    
    
    # -------------------------------------------------------------------------------------------------------------
    
    
    if ver_desarrollo >= 0:
        print("-"*200)


        print("Resumen:")
        print(f"Mayor indice de randic: {mejor_cota_randic}")
        print(f"Cota optimizada: {cota_optimizada}")


    
        print(" ")
        print("Redondeado")
        print(f"Mayor indice de randic: {round(mejor_cota_randic, 5)}")
        print(f"Cota optimizada: {round(cota_optimizada, 5)}")
        print(" ")
    
    
        # ver si la optimizada es mejor que solo con randic
        if round(cota_optimizada, 5) > round(mejor_cota_randic, 5):
            print("Los mejores pesos no son con randic")
        else:
            print("Los mejores pesos son con randic")
        
    
    # ver la energia del grafo
    energia = get_energia_grafo(G)
    
    if ver_desarrollo >= 0:
        print(" ")
        print(f"La energia del grafo es: {round(energia, 4)}")
    
    
    if devolver:
        return mejor_cota_randic, cota_optimizada, energia
    
    
    


#############################################################################333333
#################################################################################


def analisis_randic_subgrafos(grafo, ver= 1, devolver = 0):
    '''
    Toma un grafo
    Encuentra los subgrafos que maximizan el indice de randic
    
    ver indica si se ve todo el analisis
    
    devolver indica si se devuelve o no el indice de randic mayor entre subgrafos
    '''
    
    
    # tomar el indice de randic del grafo original
    indice_randic_original = get_randic(grafo)


    # tomar las aritas
    edges_grafo = list(grafo.edges)

    # ver todas las posibles combinaciones
    combinaciones_aristas = list(chain.from_iterable(combinations(edges_grafo, r) 
                                                for r in range(1, len(edges_grafo)+1)))

    # guardar los subgrafos con su indice de randic
    # en una lista de tuplas
    subgrafos_indices = []

    # iterar en la combinacion de las aristas
    for aristas in combinaciones_aristas:

        # hacerlo lista
        aristas = list(aristas)

        # constuir el subgrafo
        subgrafo_aristas = nx.Graph()

        # poner los nodos del grfo original
        subgrafo_aristas.add_nodes_from(grafo.nodes)

        # poner las aristas
        subgrafo_aristas.add_edges_from(aristas)

        # calcular el indice de randic
        randic_subgrafo_aristas = get_randic(subgrafo_aristas)

        # guardar en la lista
        subgrafos_indices.append((subgrafo_aristas, randic_subgrafo_aristas))


    # tomar solo los indices de randic de los subgrafos
    indices_randic_subgrafos = [indx_r for subgrafo, indx_r in subgrafos_indices]

    # tomar el indice de randic mayor
    mayor_randic_subgrafos = max(indices_randic_subgrafos)

    # identificar todos los subgrafos que alcanncal el mayor indice
    mejores_subgrafos = [subgrafo for subgrafo, idx_r in subgrafos_indices if np.isclose(idx_r, mayor_randic_subgrafos)]


    # ver la info
    if ver:
        print("Grafo original")
        print(f"Indice de randic: {indice_randic_original}")



        pos_grafo = pos=nx.kamada_kawai_layout(grafo)
    
    
        fig, ax = plt.subplots(figsize=(4, 4))
        nx.draw(grafo, pos= pos_grafo, with_labels=True, node_size=700, 
                        node_color='skyblue', ax=ax)
    
        plt.show()
    
        print(f"El indice de randic mayor de los subgrafos es: {mayor_randic_subgrafos}")
        print("Es alcanzado por los subgrafos:")
    
        for subgrafo in mejores_subgrafos:
    
            fig, ax = plt.subplots(figsize=(3, 3))
    
            fig.patch.set_edgecolor('black')
            fig.patch.set_linewidth(0.8)
    
            nx.draw(subgrafo, pos= pos_grafo, with_labels=True, node_size=700, 
                        node_color='skyblue', ax=ax)
    
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            
            plt.show()
            
            
        # mostrar que aristas se pueden quitar para subir el randic
        
        # tomar el randic de todo el grafo
        indice_randic_original = get_randic(grafo)
    
        
        # inicializar los colores vacios
        edge_colors = {}
        
        # iterar sobre las aristas del grafo
        for edge in edges_grafo:
    
            # considerar quitar la arista
            grafo_copia = grafo.copy()
            grafo_copia.remove_edge(*edge)
    
            # calcular nuevo randic
            nuevo_randic = get_randic(grafo_copia)
    
            # ver la diferencia con el original
            dif_randic = nuevo_randic - indice_randic_original
    
            # ponerlo como peso de esa arista
            grafo[edge[0]][edge[1]]['cambio_randic'] = dif_randic
            
            # indicar el color de la arista en el diccionario
            if dif_randic > 0:
                edge_colors[edge] = 'green'  # quitarla aumenta el randic
            elif dif_randic == 0:
                edge_colors[edge] = 'black'  # quitarla deja el randic igual
            else:
                edge_colors[edge] = 'red'  # quitarla aumenta el randic
                
                
                
        # dibujarlo
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    
        # calcular el pos del grafo
        pos_grafo = pos=nx.kamada_kawai_layout(grafo)
        
        # dibujar el grafo original
        nx.draw(grafo, pos=pos_grafo, with_labels=True, node_size=700, node_color='skyblue', ax = ax[0])
        ax[0].set_title("Grafo original")
        
        # dibujar las aristas con colores
        nx.draw(grafo, pos=pos_grafo, with_labels=True, node_size=700, node_color='skyblue',
                edge_color=[edge_colors[(u, v)] for u, v in grafo.edges()], ax = ax[1])
        ax[1].set_title("Impacto aristas")
        
        
        plt.show()
        
        
    if devolver:
        return mayor_randic_subgrafos
    
    
    
#########################################################################################




def obtener_grafos_conexos_pequeños():
    
    # tomar la lista de todos los grafos de 7 nodos o menos
    lista_grafos_completa = nx.graph_atlas_g()
    
    
    # filtrar grafos
    lista_grafos = []
    
    
    # iterar en los grafos, quitar los que no se quieran
    for grafo in lista_grafos_completa[1:]:
        
        # quitar los grafos con vertices aislados
        if any(deg == 0 for node, deg in grafo.degree()):
            # pasar, el grafo no cumple
            continue        
        
        # quitar grafos disconexos
        if nx.number_connected_components(grafo) > 1:
            # pasar, el grafo no cumple
            continue 
            
        # si no hay problemas, se mete
        lista_grafos.append(grafo)
        
        
    # devolver la lista
    return lista_grafos

################################################################################

def encontrar_matching_number(G):

    # ver el numero de nodos
    n = G.number_of_nodes()

    # el matching number u debe cumplir que
    # u <= floor(n/2)

    # ver cuanto es floor(n/2)
    max_posible_match_number = math.floor(n/2)

    # tomar las aristas de grafo
    # para despues explorar subconjuntos de las aristas
    aristas = list(G.edges)

    # explorar todo posible subconjunto de aristas
    # de cardinalidad menor o igual a floor(n/2)
    # para encontrar el matching number
    # empezar a buscar con los subconjuntos de aristas
    # de mayor cardinalidad, asi no se deben de exporar
    # todos los subconjuntos, pues se garantiza que se encuentra el mayor
    subconjuntos_aristas_candidatos = list(chain.from_iterable(combinations(aristas, r) 
                                                for r in range(max_posible_match_number, 0, -1)))

    # iterar en estos subconjuntos de aristas
    for aristas_probar in subconjuntos_aristas_candidatos:

        # transformar las aristas en un conjunto de aristas
        aristas_probar = set(aristas_probar)

        # ver si es matching 
        if nx.is_matching(G, aristas_probar):

            # se acaba de encontrar el mayor matching
            matching_number = len(aristas_probar)

            return matching_number, aristas_probar

################################################################################

# checar si un conjunto de vertices S es un vertex cover
def is_vertex_cover(G, S):
    
    # iterar en las aristas
    for edge in G.edges:
        
        # si ninguno de sus extremos esta en S
        # entonces S no es un vertex cover
        if (edge[0] not in S) and (edge[1] not in S):
            return False

    # se termina el for
    # nunca se entro al if
    # entonces
    # para toda arista, algun extremo esta en S
    return True


################################################################################

def encontrar_minimum_vertex_cover(G):

    # fuerza bruta :(

    # tomar los nodos del grafo
    nodos = list(G.nodes)

    # explorar todas las combinaciones
    # de menor a mayor
    combinaciones_nodos = list(chain.from_iterable(combinations(nodos, r) 
                                                    for r in range(1, len(nodos)+1)))

    # iterar en las combinaciones de nodos
    # notar que se hace de menor cardinalidad a mayor
    # entonces el primero encontrado es el menor
    for nodos_checar in combinaciones_nodos:

        # hacerlo conjunto
        nodos_checar = set(nodos_checar)

        # ver si es vertex cover
        if is_vertex_cover(G, nodos_checar):

            # se encontro el minimum vertex cover
            return len(nodos_checar), nodos_checar


################################################################################

def is_bipartite_complete(G):
    
    # si no es bipartita entonces
    # obvio no es bipartita completa
    if not nx.is_bipartite(G):
        return False


    # si es bipartita

    # tomar los conjuntos de nodods V1 y V2
    V1, V2 = bipartite.sets(G)

    # obtener n1 y n2
    n1 = len(V1)
    n2 = len(V2)

    # tomar el numero de aristas
    m = G.number_of_edges()

    # es bipartita completa si y solo si
    # tiene n1 x n2 aristas

    if m == n1*n2:
        return True
    else:
        return False


################################################################################


def is_bipartita_completa_balanceada(G):
    
    # si no es bipartita completa entonces
    # obvio no es bipartita completa balancead
    if not is_bipartite_complete(G):
        return False
    
    # si es bipartita completa
    
    # tomar los conjuntos de nodods V1 y V2
    V1, V2 = bipartite.sets(G)
    
    # es balanceada si tienen los mimsmo tamaños
    return len(V1) == len(V2)
    

################################################################################

def max_degree(G):
    
    # tomar diccionario de grados de todos los nodos
    degrees = dict(G.degree())

    # tomar el grado mayor
    max_degree = max(degrees.values())
    
    return max_degree



################################################################################


# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html#networkx.algorithms.cycles.simple_cycles
def get_numero_ciclos(G):
    
    
    # obtener los ciclos simples del grafo
    # trayectorias cerradas (definicion de temas 3)
    ciclos = list(nx.simple_cycles(G))

    # ver cuantos ciclos hay
    num_ciclos = len(ciclos)

    # contar numero de ciclos par e impar
    num_ciclos_pares = 0
    num_ciclos_impares = 0

    # iterar en los ciclos
    for ciclo in ciclos:

        # dado que se representan como vertices
        # sin repetir el primero y el ultimo
        # entonces el numero de nodos es el numero de aristas

        # ver si es par
        if len(ciclo)%2 == 0:
            # agregar la cuenta de ciclos pares
            num_ciclos_pares +=1

        # si no es par es impar
        else:
            # agregar la cuenta de ciclos impares
            num_ciclos_impares += 1

    # ver que se hayan considerado todos
    assert num_ciclos == num_ciclos_pares+num_ciclos_impares

    # devolver las 3 cuentas
    return num_ciclos, num_ciclos_pares, num_ciclos_impares



################################################################################

# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis
def get_numero_ciclos_base(G):
    
    
    # obtener una base de ciclos para el grafo
    # (definicion de analisis topologico)
    ciclos_base = nx.cycle_basis(G)

    # ver cuantos ciclos hay en la base
    num_ciclos_base = len(ciclos_base)

    # contar numero de ciclos par e impar en la base
    num_ciclos_pares_base = 0
    num_ciclos_impares_base = 0

    # iterar en los ciclos de la base
    for ciclo_base in ciclos_base:

        # dado que se representan como vertices
        # sin repetir el primero y el ultimo
        # entonces el numero de nodos es el numero de aristas

        # ver si es par
        if len(ciclo_base)%2 == 0:
            # agregar la cuenta de ciclos pares
            num_ciclos_pares_base +=1

        # si no es par es impar
        else:
            # agregar la cuenta de ciclos impares
            num_ciclos_impares_base += 1

    # ver que se hayan considerado todos
    assert num_ciclos_base == num_ciclos_pares_base+num_ciclos_impares_base

    # devolver las 3 cuentas
    return num_ciclos_base, num_ciclos_pares_base, num_ciclos_impares_base


################################################################################


# ver si el grafo tiene matriz de adjacencia singular
def is_singular_graph(G):
    
    # obtener la matriz de adyacencia del grafo
    A = nx.to_numpy_array(G)

    # obtener el determinante
    det = np.linalg.det(A)

    # si es singular entonces 
    # el determinante es cero
    # (muy cercano por cuestiones numericas)
    return np.isclose(det, 0)



################################################################################


# obtener una lista con las energias de cada nodo
def get_node_energy(G):
    
    # obtener la matriz de adyacencia del grafo
    A = nx.to_numpy_array(G)

    # obtener el valor absoluto de esta matriz
    abs_A = scipy.linalg.sqrtm(np.matmul(A, A.T))

    # tomar la parte real (por cuestiones numericas)
    # de la diagonal
    energia_vertices = np.real(np.diagonal(abs_A))

    return energia_vertices





################################################################################


# obtener los eigenvalores de la matriz de adj
def espectro_matriz_adj(G):
    
    # obtener el spectrum
    spectrum = nx.adjacency_spectrum(G)
    
    # tomar las partes reales (errores numericos)
    spectrum = np.real(spectrum)
    
    # redondear a 10 decimales (errores numericos)
    spectrum = np.round(spectrum, 10)
    
    return spectrum

################################################################################



def is_regular(graph):
    degrees = [deg for node, deg in graph.degree()]
    return len(set(degrees)) == 1

################################################################################

def obtener_estadisticas_grafo(grafo):
    
    # inicializar vacio
    estadisticas = dict()
    
    
    # informacion que totalmente caracteriza el grafo
    # es decir, los nodos y las aristas tal cual
    estadisticas['nodos'] = list(grafo.nodes())
    estadisticas['aristas'] = list(grafo.edges())
    
    
    # cosas basicas, n y m
    estadisticas['numero nodos'] = grafo.number_of_nodes()
    estadisticas['numero aristas'] = grafo.number_of_edges()
    
    # densidad y grado maximo
    estadisticas['densidad'] = nx.density(grafo)
    estadisticas['grado maximo'] = max_degree(grafo)
    
    # ver si es regular
    estadisticas['regular'] = is_regular(grafo)
    
    # ver si es Euleriano
    estadisticas['Euleriano'] = nx.is_eulerian(grafo)
    
    # bipartita y bipartita completa
    estadisticas['bipartita'] = nx.is_bipartite(grafo)    
    estadisticas['bipartita completa'] = is_bipartite_complete(grafo)
    estadisticas['bipartita completa balanceada'] = is_bipartita_completa_balanceada(grafo)
    
    # arbol
    estadisticas['arbol'] = nx.is_tree(grafo)
    
    # minimum vertex covering
    vertex_cover_number, _ = encontrar_minimum_vertex_cover(grafo)
    estadisticas['vertex cover number'] = vertex_cover_number
    
    # mathing number
    match_n, _ = encontrar_matching_number(grafo)
    estadisticas['matching number'] = match_n
    

    # numero de ciclos
    num_c, num_c_par, num_c_impar = get_numero_ciclos(grafo)
    estadisticas['numero ciclos'] = num_c
    estadisticas['numero ciclos pares'] = num_c_par
    estadisticas['numero ciclos impares'] = num_c_impar
    
    
    # numero de ciclos en la base
    num_c_base, num_c_par_base, num_c_impar_base = get_numero_ciclos_base(grafo)
    estadisticas['numero ciclos base'] = num_c_base
    estadisticas['numero ciclos pares base'] = num_c_par_base
    estadisticas['numero ciclos impares base'] = num_c_impar_base
    
    
    # ver si el grafo es singular
    estadisticas['singular'] = is_singular_graph(grafo)
    
    
    # poner en una lista el espectro de la matriz de adjacencia
    estadisticas['espectro adj'] = espectro_matriz_adj(grafo)
    
    
    # poner las energias de cada vertice
    estadisticas['energias vertice'] = get_node_energy(grafo)
    
    
    # finalmente, randic y energia
    estadisticas['randic'] = get_randic(grafo)
    estadisticas['mejor randic'] = analisis_randic_subgrafos(grafo, ver=0, devolver = 1)
    estadisticas['energia'] = get_energia_grafo(grafo)
    
    return estadisticas

##################################################################################
