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
import tsp_hamiltonian as H

"""
The deterministic version of the algorithm. This file contains the function that
calls the solver that breaks up any gradient tie by selecting the first option. 
"""


def parse_args():
    "Arguments parsing to make the file callable."
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--B", type=float, default=0.03)
    parser.add_argument("--it", type=int, default=1)
    return parser.parse_args()


def TSP_ver(N, fqubit, graph, path, energy, json_dic):
    "A function to verify that the path returned by the algorithm is a Hamiltonian path."
    global path_H
    path_edges = []
    for i in range(len(path)):
        path_edges.append((path[i] % (N), path[(i + 1) % len(path)] % (N)))

    path_nodes = [i for edge in path_edges for i in edge]

    if list(set(path_nodes)) == list(set(np.arange(N))):
        all_nodes = True
    else:
        all_nodes = False

    ### Is it an exiting path on the graph.
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

    ### Does each node appear twice in the edge representation of the path.
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

    ### Is the path a cycle as it should be.
    if path_nodes[0] == path_nodes[-1]:
        loop = True
    else:
        loop = False

    ### Initialize a path weight counter
    path_weight = 0

    ### If it is indeed a Hamiltonian cycle, compute the path weight.
    if all_nodes and existent_path and node_occ and loop:
        path_H = 'Is a hamiltonian path'
        for i in range(len(path_edges)):
            weight = graph[path_edges[i][0]][path_edges[i][1]]["weight"]
            path_weight += weight

    ### If not a Hamiltonian cycle, return the broken constraint.
    if not existent_path:
        path_H = 'Non-existent path'
    elif not node_occ or not all_nodes:
        path_H = 'Not all nodes occur or nodes occur more than twice'
    elif not loop:
        path_H = 'Not a solution to TSP'

    json_dic["fqubit_{}".format(fqubit)] = {}

    json_dic["fqubit_{}".format(fqubit)]["is_H_path"] = path_H

    json_dic["fqubit_{}".format(fqubit)]["Path"] = path_edges

    json_dic["fqubit_{}".format(fqubit)]["energy"] = energy

    json_dic["fqubit_{}".format(fqubit)]["path_weight"] = path_weight
    return json_dic


def solve(json_dic, N, B, graph, fqubit):
    "The function calling the Adapt-Clifford algorithm to solve the TSP instance."

    ### Compensate for indexation error in the results saving.
    fqubit = fqubit - 1

    ### Empty matrix to be filled with the edge weights and the Hamiltonian coefficients.
    W = np.zeros(shape=(N ** 2 + 1, N ** 2 + 1), dtype=float)

    ### Computing the Hamiltonian coefficients.
    W = H.adapt_tsp_hamiltonian(N, B, W, graph)
    W = np.round_(W, decimals=6)

    ### System parameters
    nbit = N ** 2 + 1

    ### Generate all possible qubit combinations
    combis = [combi for combi in itertools.combinations(range(nbit), 2)]
    wvec = af.weights_vector(nbit, combis, W)

    ### Create Hamiltonian terms
    Hterms = af.hamil_terms(nbit, combis, "Z")

    ### Build the initial empty tableau and add the layer of Hadamards
    circuit = stim.TableauSimulator()
    af.add_H_layer(nbit, circuit)

    ### Flip the state of qubit at position fqubit
    circuit.z(fqubit)

    ### Initialize the records of active and inactive qubits
    active_qubits_k = []
    active_qubits_j = []
    inactive_qubits = list(range(nbit))

    gate_posis = []

    for nn in range(nbit - 1):
        ##-- find the combinations corresponding to only the active qubits
        if nn == 0:
            qpair = np.random.choice(W[:, fqubit].nonzero()[0])

            gra = W[qpair, fqubit]
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
        if sol_string[zz] == True:
            set1.append(zz)
        else:
            set2.append(zz)

    #
    path = min([set1, set2], key=len)
    result_dictionary = TSP_ver(N, fqubit, graph, path, energy, json_dic)

    return result_dictionary


def main():
    start = time.time()
    number_of_cpus = cpu_count()

    args = parse_args()

    N = args.N

    d = args.d

    B = args.B

    it = args.it

    json_dic = {}

    filename = "./graph_generation/graph_instances/N_{}/d_{}/N_{}_d_{}_graph_instances.json".format(N, d, N, d)
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    adj_mat = data[it - 1]

    graph = json_graph.adjacency_graph(adj_mat["{}".format(it)])

    tsp_results = Parallel(n_jobs=int(number_of_cpus))(
        delayed(solve)(json_dic, N, B, graph, fqubit) for fqubit in range(1, N + 1))

    with open(
            "./graph_generation/graph_instances/N_{}/d_{}/B_{}/N_{}_d_{}_B_{}_it_{}_tsp_test_results.json".format(N, d,
                                                                                                                  B, N,
                                                                                                                  d, B,
                                                                                                                  it),
            "w") as outfile:
        json.dump(tsp_results, outfile)
    end = time.time()

    total_time = end - start
    print("\n" + str(total_time))


if __name__ == "__main__":
    main()
