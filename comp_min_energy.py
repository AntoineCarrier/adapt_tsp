import numpy as np
from networkx.readwrite import json_graph
import itertools
import json
import os
import adapt_functions as af
import tsp_hamiltonian as H

"""
A file to compute the minimal energy of the hamiltonian on standard graph instances
for the algorithm< approximation ratios
"""

json_dic = {}
### The standard instances with skewed weight distribution.
filename = "./adapt_test_results/skewed_w_dist_gi_4_12.json"
if os.path.exists(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

for N in [5]:
    json_dic["N_{}".format(N)] = {}
    for skewness in np.arange(-1.5, 2.0, 0.5):
        skewness = np.round(skewness, 1)

        json_dic["N_{}".format(N)]["s_{}".format(skewness)] = {}
        adj_mat = data["N_{}".format(N)]["s_{}".format(skewness)]

        for B in [0.04]:
            B = np.round(B, 2)

            json_dic["N_{}".format(N)]["s_{}".format(skewness)]["B_{}".format(B)] = []

            for it in range(0, 101):
                ### Read the adjacency matrix in the .json as a networkx grapj
                graph = json_graph.adjacency_graph(adj_mat["it_{}".format(it)])

                ### Build a matrix of zeros to be filled with the Hamiltonian coefficients
                W = np.zeros(shape=(N ** 2 + 1, N ** 2 + 1), dtype=float)

                ### Compute the Hamiltonian coeficient
                W = H.adapt_tsp_hamiltonian(N, B, W, graph)

                ### Generate every possible qubit combination
                combis = [combi for combi in itertools.combinations(range(N ** 2 + 1), 2)]

                wvec = af.weights_vector(N ** 2 + 1, combis, W)

                ### Compute the minimum energy from the coefficient's matrix
                min_ener = np.min(af.hamil(combis, N ** 2 + 1, W))

                json_dic["N_{}".format(N)]["s_{}".format(skewness)]["B_{}".format(B)].append(min_ener)

    with open("./N_{}_comp_st_inst_min_ener.json".format(N), "w") as outfile:
        json.dump(json_dic, outfile)
