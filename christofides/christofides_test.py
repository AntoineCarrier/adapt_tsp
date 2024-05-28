import christofides
from networkx.readwrite import json_graph
import networkx as nx
import numpy as np
import os
import json
from joblib import Parallel, delayed, cpu_count
import time

start = time.time()
number_of_cpus = cpu_count()

filename = "./skewed_w_dist_gi_4_12.json"
if os.path.exists(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

for N in [4, 5, 6, 8, 10, 12]:
    for s in np.arange(-2.0, 2.0, 0.5):
        s = np.round(s, 1)

        adj_mat = data["N_{}".format(N)]["s_{}".format(s)]
        for it in range(0, 101):

            graph = json_graph.adjacency_graph(adj_mat["it_{}".format(it)])

            distance_matrix = nx.to_numpy_array(graph)
            distance_matrix = distance_matrix.astype(int)
            json_dic = {}
            tsp_results = christofides.tsp(distance_matrix)
            with open("./chris_test_results/N_{}_s_{}_it_{}_christofides_test_results.json".format(N, s, it), "w") as outfile:
                json.dump(tsp_results, outfile)
end = time.time()
print(end - start)

#with open("N_4_B_0.03_s_-0.5_it_0_red_tsp_test_results.json", 'r') as file:
    #data = json.load(file)
#graph = json_graph.adjacency_graph(data['N_4']['s_-2.0']['it_0'])


#print(graph)
# distance_matrix = [[0,45,65,15],[45,0,56,12],[65,56,0,89],[15,12,89,0]]

#distance_matrix = nx.to_numpy_array(graph)
#distance_matrix = distance_matrix.astype(int)

#print(distance_matrix)


#TSP = christofides.tsp(distance_matrix)

#print(TSP)
