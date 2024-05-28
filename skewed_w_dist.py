import numpy as np

import networkx as nx 

from scipy.stats import skewnorm

import json


json_dic = {}

for N in [24, 26, 28, 30]:
    utri_indices = (N*(N-1))/2

    l_indices = np.tril_indices(N, k=-1)

    x = np.linspace(0, int(utri_indices), int(utri_indices)) 

    arr = np.zeros((N, N))

    json_dic["N_{}".format(N)] = {}

    for s in np.arange(-2.0, 2.0, 0.5):

        s = np.round(s, 1)
        json_dic["N_{}".format(N)]["s_{}".format(s)]={}
        
        for it in range(0, 101):
        
            w = skewnorm.rvs(s, 3, 5, size=int(utri_indices))+10  

            arr[l_indices] = w

            adj_mat = arr + arr.T

            adj_mat = np.array(adj_mat)

            G = nx.from_numpy_array(adj_mat)

            adj_data = nx.adjacency_data(G)

            json_dic["N_{}".format(N)]["s_{}".format(s)]["it_{}".format(it)] = adj_data

            print(json_dic.keys())

with open("skewed_w_dist_gi_24_30.json", "w") as outfile:
    json.dump(json_dic, outfile)
