
import numpy as np
from math import log
from openfermion import (
    QubitOperator,
    get_sparse_operator
)
import networkx as nx

#
#    Functions to convert factorized format ("factorization_dict") to length 2^N vector format for quantum states
#

def expand_tensor_product(factorization_dict, Nqubits):
    labels          = 'abcdefghijklmnopqrstuvwxyz'
    tensor_dict     = dict([(''.join([labels[x] for x in k]), np.reshape(v, [2] * len(k))) for (k,v) in factorization_dict.items()])
    einsum_equation = ','.join([x for x in tensor_dict.keys()]) + '->' + labels[:Nqubits]
    psi_tensor      = np.einsum(einsum_equation, *tensor_dict.values())
    psi             = np.reshape(psi_tensor, 2 ** Nqubits)

    return psi

def expand_tensor_product_for_incomplete_qubit_set(factorization_dict):
    labels          = 'abcdefghijklmnopqrstuvwxyz'
    tensor_dict     = dict([(''.join([labels[x] for x in k]), np.reshape(v, [2] * len(k))) for (k,v) in factorization_dict.items()])
    einsum_equation = ','.join([x for x in tensor_dict.keys()]) + '->' + ''.join(sorted(''.join([x for x in tensor_dict.keys()])))
    psi_tensor      = np.einsum(einsum_equation, *tensor_dict.values())
    psi             = np.reshape(psi_tensor, 2 ** len(''.join(sorted(''.join([x for x in tensor_dict.keys()])))))

    return psi

#
#    Functions to obtain unified factorization of bra and ket given maximal factorizations
#

def partition_from_dict(factorization_dict):
    return list(factorization_dict.keys())

def obtain_join_graph(partA, partB):
    all_vals = {val for block in partA + partB for val in block}

    G = nx.Graph()
    G.add_nodes_from(all_vals)

    for block in partA + partB:

        for i in block:
            for j in block:
                if j > i:
                    G.add_edge(i, j)

    return G

def obtain_join_partition(partA, partB):
    G   = obtain_join_graph(partA, partB)
    ccs = nx.connected_components(G)
    
    return [tuple(sorted(cc)) for cc in ccs]

def obtain_coarse_partitioning(join_partition, factorization_dict):
    coarse_dict = {}

    for block in join_partition:
        sub_dict           = {k : v for (k, v) in factorization_dict.items() if set(k).issubset(set(block))}
        psi                = expand_tensor_product_for_incomplete_qubit_set(sub_dict)
        coarse_dict[block] = psi

    return coarse_dict

def obtain_coarse_dicts(factorization_dict1, factorization_dict2, N):
    partition1 = partition_from_dict(factorization_dict1)
    partition2 = partition_from_dict(factorization_dict2)

    join_partition = obtain_join_partition(partition1, partition2)

    coarse1 = obtain_coarse_partitioning(join_partition, factorization_dict1)
    coarse2 = obtain_coarse_partitioning(join_partition, factorization_dict2)

    return join_partition, coarse1, coarse2

#
#    Functions to label qubits (SV, SW, SN) and factors of unified factorization (C, Q)
#

def QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition):
    QC_dictionary = {}

    for block in join_partition:
        assignment = 'C'
        for i in block:
            if bra_labels[i] == 'W' or ket_labels[i] == 'W':
                assignment = 'Q'
                break
        QC_dictionary[block] = assignment
    
    return QC_dictionary

#
#    Functions to obtain partially evaluated Hamiltonian matrix elements based on quantum/classical block assignments and unified-factorizations
#

def split_pauli_operator(T, C):
    
    if isinstance(T, QubitOperator):
        T = list(T.terms.keys())[0]

    TC = tuple([term for term in T if term[0] in C])
    TR = tuple([term for term in T if not term[0] in C])

    return TC, TR

def relabel_qubits_in_pauli_term(TC, C):
    label_reassignment = {x : i for i, x in enumerate(C)}
    new_terms_tuple    = []
    for term in TC:
        new_term = (label_reassignment[term[0]], term[1])
        new_terms_tuple.append(new_term)
    return tuple(new_terms_tuple)

def evaluate_matrix_element_given_term(psi, phi, TC):
    Nqubits      = int(log(len(psi), 2))
    Pauli_sparse = get_sparse_operator(QubitOperator(TC), Nqubits)
    return (psi @ Pauli_sparse @ phi.T)[0,0]

def partially_evaluate_pauli_term(psi, phi, T, C):
    TC, TR       = split_pauli_operator(T, C)
    TCrelabelled = relabel_qubits_in_pauli_term(TC, C)
    return evaluate_matrix_element_given_term(psi, phi, TCrelabelled) * QubitOperator(TR)

def partially_evaluate_hamiltonian_matrix_element(psi, phi, H, C):
    H_evaluated = QubitOperator()
    for term, coef in H.terms.items():
        H_evaluated += coef * partially_evaluate_pauli_term(psi, phi, term, C)
    return H_evaluated

def evaluate_fully_classical_factors(factorization_dict_bra, factorization_dict_ket, bra_labels, ket_labels, H, Nqubits):

    join_partition, coarse_dict_bra, coarse_dict_ket = obtain_coarse_dicts(factorization_dict_bra, factorization_dict_ket, Nqubits)
    QC_assignment_dict                               = QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition)

    full_Q_block = set()
    for block, assignment in QC_assignment_dict.items():
        if assignment == 'C':
            H = partially_evaluate_hamiltonian_matrix_element(coarse_dict_bra[block], coarse_dict_ket[block], H, block)
            del coarse_dict_bra[block]
            del coarse_dict_ket[block]

        else:
            full_Q_block.update(block)

    full_Q_block = sorted(tuple(full_Q_block))            
    Heff         = QubitOperator()
    for term, coef in H.terms.items():
        Heff += coef * QubitOperator(relabel_qubits_in_pauli_term(term, full_Q_block))

    Qstate_bra = expand_tensor_product_for_incomplete_qubit_set(coarse_dict_bra)
    Qstate_ket = expand_tensor_product_for_incomplete_qubit_set(coarse_dict_ket)

    return Heff, Qstate_bra, Qstate_ket


