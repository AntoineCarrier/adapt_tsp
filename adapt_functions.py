import numpy as np
import stim


###
#   Some utility functions
###

### Add a layer of Hadamars to the Tableau
def add_H_layer(nbit, c: stim.TableauSimulator):
    """
    This function adds a layer of Hadamar gates to the TableauSimulator
    
    nbit: number of qubits/nodes in the graph
    """
    c.h(*range(nbit))
#

### weight matrix for a complete graph
def weight_matrix_complete(nbit):
    mat = np.zeros((nbit, nbit))
    
    for ii in range(nbit):
        for jj in range(nbit):
            if jj > ii: mat[ii, jj] = np.random.random()
        #
    #
    return mat + mat.T
#

### compute the expectation value of the energy
def hamil_terms(nbit, combis, term):
    """
    This function constructs all the ZZ Hamiltonian terms
    
    nbit: number of qubits/nodes 
    combis: a list of tuples (l,m) indicating the edges where the Hamiltonian acts nontrivially  
    term: either X, Y, or Z 
    
    returns a list of the 
    """
    terms = []
    for combi in combis:
        pstring = stim.PauliString(nbit)
        pstring[combi[0]] = term
        pstring[combi[1]] = term
        
        terms.append(pstring)
    #
    return terms
#

def weights_vector(nbit, combis, W):
    """
    For a weighted graph with adjacency matrix W, this function vectorizes W following the ordering of the 
    list of edges given by combis 
    
    nbit: number of qubits/nodes 
    combis: a list of tuples (l,m) indicating the edges where the Hamiltonian acts nontrivially 
    """
    weights = []
    for combi in combis:
        weights.append(W[combi])
    #
    return weights
#

def hamil_expectation_vals(terms, c: stim.TableauSimulator):
    """
    This function computes the expectation value of all of the 
    Hamiltonian terms in the current state
    
    terms: a list with all the Hamiltonian terms, each given by a PauliString
    c: the current state ecoded as a TableauSimulator
    """
    
    vals = []
    for term in terms:
        val = c.peek_observable_expectation(term)
        vals.append(val)
    #
    return vals
#

def current_energy(weights, hterms, c: stim.TableauSimulator):
    """
    This function computes the mean energy of the current state
    
    weights: the vector of edge weights. They follow the same ordering as the edge list
    hterms: the vector of Hamiltonian terms
    """
    expects = hamil_expectation_vals(hterms, c)
    return np.dot(weights, expects)
#


### construct one term of the Hamiltonian
def hamiltonian_term(combi, nbit, W):
    """
    This function constructs a 2^nbit vector corresponding to the energy of one of
    the Hamiltonian terms 
    
    combis: A list of tuples representing the edges 
    nbit: the number of qubits/nodes in the graph
    W: the adjacency matrix of the weighted graph
    """
    mat = None
    if nbit-1 == combi[1] and nbit-2 != combi[0]:
        mat = np.kron([1,1], [1,-1])
    elif nbit-2 == combi[1]:
        mat = np.kron([1,-1], [1,1])
    elif nbit-1 == combi[1] and nbit-2 == combi[0]:
        mat = np.kron([1,-1], [1,-1])
    else:
        mat = np.kron([1,1], [1,1])

    for nn in range(nbit-3, -1, -1):
        if nn in combi:
            mat = np.kron([1,-1], mat)
        else:
            mat = np.kron([1,1], mat)
        #
    #
    return W[combi]*np.array(mat)
#


### create the full Hamiltonian
def hamil(combis, nbit, W):
    """
    This function returns a vector containing all the energies of the problem Hamiltonian
    
    combis: A list of tuples representing the edges 
    nbit: the number of qubits/nodes in the graph
    W: the adjacency matrix of the weighted graph
    """
    hamil = np.zeros(2**nbit)
    
    for combi in combis:
        hamil = hamil + hamiltonian_term(combi, nbit, W)
    #
    return hamil
#


###
#    Utilty functions to build the gate pool
###

### adding the YZ gate 
def add_YZ_gate(q1, q2, c: stim.TableauSimulator):
    """
    This function applies a e^(pi/4 YZ) gate to the state 
    
    q1: the index of the first qubit 
    q2: the index of the second qubit 
    c: the current state in the form of a TableauSimulator
    """
    
    c.s_dag(q1)
    c.h(q2)
    c.cnot(q1, q2)
    c.z(q1)
    c.h_yz(q1)
    c.cnot(q1, q2)
    c.s(q1)
    c.h(q2)
#


###
#    Utility functions to compute the gradients
###
def gradient(inaqubit, W, aqubits_k, aqubits_j, c: stim.TableauSimulator):
    """
    Given an inactive qubit, this function computes the gradient of it with
    with respect to the two qubits defining the bipartition.
    
    inaqubit: the inactibe qubit under consideration 
    W: a matrix of the edge weights
    aqubits_k: the active qubits which were entangled with qubit k
    aqubits_j: the active qubits which were entangled with qubitt j
    c: the current state encoded as a TableauSimulator
    """
    
    lindex_k = np.intersect1d(np.nonzero(W[:,inaqubit])[0], aqubits_k)
    lindex_j = np.intersect1d(np.nonzero(W[:,inaqubit])[0], aqubits_j)
    
    sum_weights_k = np.sum(W[ll, inaqubit] for ll in lindex_k)
    sum_weights_j = np.sum(W[ll, inaqubit] for ll in lindex_j)
    
    grad_k = -sum_weights_k + sum_weights_j
    return grad_k

#
def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j, c: stim.TableauSimulator):
    """
    This function finds the inactive qubit b with the largest gradient, and the 
    corresponding initial qubit, k or j, with which this largest gradient occurs
    
    inaqubits: the vector of inactive qubits
    W: a matrix of the edge weights
    aqubits_k: the active qubits which were entangled with qubit k
    aqubits_j: the active qubits which were entangled with qubitt j
    c: the current state encoded as a TableauSimulator
    """
    
    all_grads_k = [gradient(inaqubit, W, aqubits_k, aqubits_j, c) for inaqubit in inaqubits]
    all_grads_j = -1.0*np.array(all_grads_k)
    
    pos_max_k = np.argmax(all_grads_k)
    pos_max_j = np.argmax(all_grads_j)
    
    if all_grads_k[pos_max_k] > all_grads_j[pos_max_j]:
        return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
    elif all_grads_k[pos_max_k] < all_grads_j[pos_max_j]:
        return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j] 
    else:
        char = np.random.choice([1,2])
        if char == 1:
            return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
        elif char == 2:
            return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
        #
    #
#

