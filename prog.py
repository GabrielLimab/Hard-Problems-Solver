import networkx as nx
import sys
import numpy as np
import random
import heapq
import math
import time
import signal
import csv
import tracemalloc
from networkx.algorithms import tree

instances = sys.argv[1]
distanceFunction = sys.argv[2]
algorithm = sys.argv[3]
instances = int(instances)
instances = 2**instances

pointsList = []
for i in range(instances):
    pointsList.append((random.randint(1,100),random.randint(1,100)))

distancesMatrix = np.zeros((instances,instances),dtype=int)

def calculateDistance(pointsList,distancesMatrix,distanceFunction,instances):
    if distanceFunction == "euclidian":
        for i in range(instances):
            for j in range(instances):
                firstPoint = pointsList[i]
                secondPoint = pointsList[j]
                distance = abs(firstPoint[0]-secondPoint[0]) + abs(firstPoint[1]-secondPoint[1])
                distance = math.sqrt(distance)
                distancesMatrix[i][j] = distance
    elif distanceFunction == "manhattan":
        for i in range(instances):
            for j in range(instances):
                firstPoint = pointsList[i]
                secondPoint = pointsList[j]
                distance = abs(firstPoint[0]-secondPoint[0]) + abs(firstPoint[1]-secondPoint[1])
                distancesMatrix[i][j] = distance
    return distancesMatrix

distancesMatrix = calculateDistance(pointsList,distancesMatrix,distanceFunction,instances)

G = nx.complete_graph(instances)

for (u,v) in G.edges():
    G.edges()[u,v]['weight'] = distancesMatrix[u][v]
    
A = G.adj   

tsp = nx.approximation.traveling_salesman_problem
appSolution = tsp(G)
appWeight = 0
for i in range(instances):
    appWeight += A[appSolution[i]][appSolution[i+1]]['weight']

def findMinEdge(A,origin,n,path,level):
    min = float('inf')
    if level == 1:
        for i in range(n):
            if origin!=i and A[origin][i]['weight'] < min:
                min = A[origin][i]['weight']
                minEdge = (origin,i)
    else:
        for i in range(level):
            if path[i]==origin and i==0:
                minEdge = (origin,path[i+1])
                return minEdge
            elif path[i]==origin and i==len(path)-1:
                minEdge = (origin,path[i-1])
                return minEdge
            elif path[i] == origin and i!=0 and i != len(path)-1:
                minEdge = (path[i-1],origin)
                return minEdge
        minEdge = findMinEdge(A,origin,n,path,1)
        return minEdge
    return minEdge

def findSecondMinEdge(A,origin,n,minEdge,path,level):
    min = float('inf')
    if level == 1:
        for i in range(n):
            if origin != i and A[origin][i]['weight'] < min and minEdge[1] != i:
                min = A[origin][i]['weight']
                secondMinEdge = (origin,i)
    else:  
        for i in range(level):
            if path[i] == origin and i==0:
                secondMinEdge = findSecondMinEdge(A,origin,n,minEdge,path,1)
                return secondMinEdge
            elif path[i] == origin and i == len(path)-1:
                secondMinEdge = findSecondMinEdge(A,origin,n,minEdge,path,1)
                return secondMinEdge
            elif path[i] == origin and i!=0 and i != len(path)-1:
                secondMinEdge = (path[i+1],origin)
                return secondMinEdge

        secondMinEdge = findSecondMinEdge(A,origin,n,minEdge,path,1)
        return secondMinEdge

    return secondMinEdge

def bound(A,n,path,level):
    bound = 0
    if level == 1:
        for i in range(n):
            firstMin = findMinEdge(A,i,n,path,level)
            secondMin = findSecondMinEdge(A,i,n,firstMin,path,level)
            bound += A[firstMin[0]][firstMin[1]]['weight'] + A[secondMin[0]][secondMin[1]]['weight']
        bound = math.ceil(bound/2)
    else:
        for i in range(n):
            firstMin = findMinEdge(A,i,n,path,level)
            secondMin = findSecondMinEdge(A,i,n,firstMin,path,level)
            bound += A[firstMin[0]][firstMin[1]]['weight'] + A[secondMin[0]][secondMin[1]]['weight']
        bound = math.ceil(bound/2)
    return bound

def branchAndBound(A,n):
    # Definição da raiz do algoritmo
    root = (bound(A,n,[0],1),1,0,[0])
    queue = []
    # Inicialização do Heap
    heapq.heapify(queue)
    heapq.heappush(queue,root)
    best = float('inf')
    sol = []
    # Para cada elemento da heap calcular a melhor solução possível e comparar com a melhor até encontrar a melhor
    while len(queue) != 0:
        node = heapq.heappop(queue)
        nodeBound = node[0]
        level = node[1]
        cost = node[2]
        path = node[3]
        if level > n:
            if best > cost:
                best = cost
                sol = path
        elif nodeBound < best:
            if level < n:
                for k in range(n):
                    if k not in path and path[-1] != k and bound(A,n,path+[k],level+1) < best:
                        heapq.heappush(queue,(bound(A,n,path+[k],level+1),level+1,cost+A[path[-1]][k]['weight'],path+[k]))
            elif path[-1] != 0 and bound(A,n,path+[0],level+1) < best:
                heapq.heappush(queue,(bound(A,n,path+[0],level+1),level+1,cost+A[path[-1]][0]['weight'],path+[0]))
    return sol,best

def twiceAroundTheThree(G,A,n):
    weight = 0
    # Árvore Geradora Mínima do grafo
    mst = tree.minimum_spanning_tree(G,algorithm="prim")
    # Lista de vértices ordenadas pela pré-ordem de visitas na árvore
    nodesList = list(nx.dfs_preorder_nodes(mst, source=0))
    nodesList = nodesList + [0]
    # Circuito hamiltoniano
    for i in range(instances):
        weight += A[nodesList[i]][nodesList[i+1]]['weight']
    return nodesList,weight

def christofides(G,A,n):
    weight = 0
    # Árvore Geradora Mínima do grafo
    mst = tree.minimum_spanning_tree(G,algorithm="prim")
    # Vértices com grau ímpar
    mstEdges = mst.edges
    mstDegrees = np.zeros(len(mstEdges)+1)
    for i in mstEdges:
        mstDegrees[i[0]] += 1
        mstDegrees[i[1]] += 1
    oddDegree = []
    for i in range(len(mstDegrees)):
        if mstDegrees[i]%2==1:
            oddDegree.append(i)
    # Subgrafo induzido a partir dos vértices de grau ímpar
    induced_subgraph = nx.Graph(G.subgraph(oddDegree))
    # Matching perfeito de peso mínimo
    M = nx.min_weight_matching(induced_subgraph)
    # Multigrafo formado com os vértices da árvore e as arestas do matching
    G2 = nx.MultiGraph(mst)
    for i in range(len(M)):
        n1 = list(M)[i][0]
        n2 = list(M)[i][1]
        G2.add_edge(n1,n2, weight=A[n1][n2]['weight'])
    # Circuito euleriano e retirada dos vértices repetidos
    sol = []
    visited = set()

    for u,v in nx.eulerian_circuit(G2):
        if not u in visited:
            sol += [u]
            visited.add(u)
    sol += [0]
    for i in range(instances):
        weight += A[sol[i]][sol[i+1]]['weight']

    return sol,weight


def signal_handler(signum,frame):
    raise Exception("Timed out!")

signal.signal(signal.SIGALRM, signal_handler)
signal.alarm(1800)

tests = open('tests.csv','a')

writer = csv.writer(tests)

def algorithmChoice(algorithm):
    if algorithm == "bnb":
        try:
            ini = time.time()
            tracemalloc.start()
            sol,best = branchAndBound(A,instances)
            fim = time.time()
            memexp = tracemalloc.get_traced_memory()[1]
            timeexp = round(fim-ini,3)
            if (best/appWeight) < 1:
                quality = (1 - (best/appWeight))*100 + 100
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality] 
            else:
                quality = 100 - ((best/appWeight) - 1)*100 
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality]
            writer.writerow(algData)
            tracemalloc.stop()
            tests.close()   
        except Exception:
            algData = [algorithm,instances,distanceFunction,'NA','NA','NA']
            writer.writerow(algData)
            tests.close()
    elif algorithm == "tat":
        try:
            ini = time.time()
            tracemalloc.start()
            sol,best = twiceAroundTheThree(G,A,instances)
            fim = time.time()
            memexp = tracemalloc.get_traced_memory()[1]
            timeexp = round(fim-ini,3)
            if (best/appWeight) < 1:
                quality = (1 - (best/appWeight))*100 + 100
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality] 
            else:
                quality = 100 - ((best/appWeight) - 1)*100 
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality] 
            writer.writerow(algData)
            tracemalloc.stop()
            tests.close()
        except Exception:
            algData = [algorithm,instances,distanceFunction,'NA','NA','NA']
            writer.writerow(algData)
            tests.close()
    elif algorithm == "chr":
        try:
            ini = time.time()
            tracemalloc.start()
            sol,best = christofides(G,A,instances)
            fim = time.time()
            memexp = tracemalloc.get_traced_memory()[1]
            timeexp = round(fim-ini,3)
            if (best/appWeight) < 1:
                quality = (1 - (best/appWeight))*100 + 100
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality] 
            else:
                quality = 100 - ((best/appWeight) - 1)*100 
                algData = [algorithm,instances,distanceFunction,timeexp,memexp,quality]
            writer.writerow(algData)
            tracemalloc.stop()
            tests.close()
        except Exception:
            algData = [algorithm,instances,distanceFunction,'NA','NA','NA']
            writer.writerow(algData)
            tests.close()

algorithmChoice(algorithm)

