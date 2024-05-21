import christofides
from networkx.readwrite import json_graph
import networkx as nx
import json


with open("skewed_w_dist_gi_4_12.json", 'r') as file:
    data = json.load(file)
graph = json_graph.adjacency_graph(data['N_4']['s_-2.0']['it_0'])


print(graph)
# distance_matrix = [[0,45,65,15],[45,0,56,12],[65,56,0,89],[15,12,89,0]]

distance_matrix = nx.to_numpy_array(graph)
distance_matrix = distance_matrix.astype(int)

print(distance_matrix)


TSP = christofides.tsp(distance_matrix)

print(TSP)
