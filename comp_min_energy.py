import numpy as np
from networkx.readwrite import json_graph
import itertools
import json
import os
import adapt_functions as af
import tsp_hamiltonian as H

json_dic = {}
filename = "./adapt_test_results/skewed_w_dist_gi_4_12.json"
if os.path.exists(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
for N in [5]:
    json_dic["N_{}".format(N)] = {}
    for s in np.arange(-1.5, 2.0, 0.5):
        s = np.round(s, 1)
        json_dic["N_{}".format(N)]["s_{}".format(s)] = {}
        adj_mat = data["N_{}".format(N)]["s_{}".format(s)]
        for B in [0.04]:
            B = np.round(B, 2)

            json_dic["N_{}".format(N)]["s_{}".format(s)]["B_{}".format(B)] = []

            for it in range(0, 101):

                graph = json_graph.adjacency_graph(adj_mat["it_{}".format(it)])

                W = np.zeros(shape=(N ** 2 + 1, N ** 2 + 1), dtype=float)
                W = H.adapt_tsp_hamiltonian(N, B, W, graph)

                combis = [combi for combi in itertools.combinations(range(N**2+1), 2)]
                wvec = af.weights_vector(N**2+1, combis, W)

                min_ener = np.min(af.hamil(combis, N**2+1, W))

                json_dic["N_{}".format(N)]["s_{}".format(s)]["B_{}".format(B)].append(min_ener)
    with open("./N_{}_comp_st_inst_min_ener.json".format(N), "w") as outfile:
        json.dump(json_dic, outfile)
