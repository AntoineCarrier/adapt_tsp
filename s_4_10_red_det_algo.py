import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools 
import stim
import argparse
from joblib import Parallel, delayed, cpu_count
import json
import os
import time

import adapt_functions as af
import reduced_tsp_hamiltonian as H

def TSP_ver(N, fqubit, graph, path, energy, json_dic):
    path_edges = []
    for i in range(len(path)):
        path_edges.append((path[i]%(N), path[(i+1)%len(path)]%(N)))

    path_nodes = [i for edge in path_edges for i in edge]


    if list(set(path_nodes)) == list(set(np.arange(N))):
        all_nodes = True
    else:
        all_nodes = False

    existence = []
    for edge in path_edges:
        if (edge[0], edge[1]) in graph.edges or (edge[1], edge[0]) in graph.edges:
            existence.append(True)
        else:
            existence.append(False)

    if all(existence):
        existent_path = True
    else:
        existent_path = False

    node_occurence = []
    for node in path_nodes:
        if path_nodes.count(node) == 2:
            node_occurence.append(True)
        else:
           node_occurence.append(False)

    if all(node_occurence) == True:
        node_occ = True
    else:
        node_occ = False

    if path_nodes[0] == path_nodes[-1]:
        loop = True
    else:
        loop = False

    
    path_weight = 0
    if all_nodes and existent_path and node_occ and loop:
        path_H = 'Is a hamiltonian path'
        for i in range(len(path_edges)):
          weight = graph[path_edges[i][0]][path_edges[i][1]]["weight"]
          print(weight)
          path_weight += weight 
    if not existent_path:
        path_H = 'Non-existent path'
    elif not node_occ or not all_nodes:
        path_H = 'Not all nodes occur or nodes occur more than twice'
    elif not loop:
        path_H = 'Not a solution to TSP'

    print(path_H)
    json_dic["fqubit_{}".format(fqubit)] = {}

    json_dic["fqubit_{}".format(fqubit)]["is_H_path"] = path_H

    json_dic["fqubit_{}".format(fqubit)]["Path"] = path_edges

    json_dic["fqubit_{}".format(fqubit)]["energy"] = energy
  
    json_dic["fqubit_{}".format(fqubit)]["path_weight"] = path_weight

    return json_dic




### Coefficients describing the couplings between nearest neighbors
def solve(json_dic, N, B, graph, fqubit):

    fqubit = fqubit-1

    W = np.zeros(shape = (N**2 + 1, N**2 + 1), dtype = float)

    W = H.adapt_tsp_hamiltonian(N, B, W, graph, fqubit)

    W = np.round_(W, decimals = 6)


    ### system parameters
    nbit = N**2 + 1

    combis = [combi for combi in itertools.combinations(range(nbit), 2)]
    wvec = af.weights_vector(nbit, combis, W)

    ### create Hamiltonian terms
    Hterms = af.hamil_terms(nbit, combis, "Z")

    ### build the initial empty tableau and add the layer of Hadamards
    circuit = stim.TableauSimulator()
    af.add_H_layer(nbit, circuit)

    ### flip the state of qubit at position fqubit
    circuit.z(fqubit)

    ### initialize the records of active and inactive qubits
    active_qubits_k = []
    active_qubits_j = []
    inactive_qubits = list(range(nbit))

    aratio = np.zeros(nbit)

    gate_posis = []

    #aratio = np.zeros(nbit)
    for nn in range(nbit-1):
        ##-- find the combinations corresponding to only the active qubits
        if nn == 0:
            qpair = np.argmax(W[:,fqubit])
            # qpair = np.random.choice(W[:,fqubit].nonzero()[0])
        
            gra = W[qpair,fqubit]
            qubits, grad = (fqubit, qpair), gra
        
            ##-- updating the records of active and inactive qubits
            active_qubits_j.append(qpair)
            active_qubits_k.append(fqubit)
        
            inactive_qubits.remove(qpair)
            inactive_qubits.remove(fqubit)
        
        else:
            aset, qpair, gra = af.pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j, circuit)
        
            if aset == "k":
                qubits = (qpair, fqubit)
                active_qubits_k.append(qpair)
            elif aset == "j":
                qubits = (qpair, active_qubits_j[0])
                active_qubits_j.append(qpair)
            #
            inactive_qubits.remove(qpair)
        #

        ##-- apply the corresponding gate to the state
        af.add_YZ_gate(qubits[0], qubits[1], circuit)
        gate_posis.append(qubits)
    
        energy = af.current_energy(wvec, Hterms, circuit)


    sol_string = circuit.measure_many(*range(nbit))
    set1, set2 = [], []
    for zz in range(nbit):
        if sol_string[zz] == True: set1.append(zz)
        else: set2.append(zz)

#
    path = min([set1, set2], key = len)
    result_dictionary = TSP_ver(N, fqubit, graph, path, energy, json_dic)

    return result_dictionary


def main():
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

                for B in np.arange(0.03, 0.06, 0.01):
                    B = np.round(B, 2)
                    json_dic = {}
                    tsp_results = Parallel(n_jobs = int(number_of_cpus))(delayed(solve)(json_dic, N, B, graph, fqubit) for fqubit in range(1, N+1))

                    with open("./N_{}_B_{}_s_{}_it_{}_red_tsp_test_results.json".format(N, B, s, it), "w") as outfile:
                        json.dump(tsp_results, outfile)
    end = time.time()

    total_time = end - start
    print("\n"+ str(total_time))
if __name__== "__main__":
    main()
