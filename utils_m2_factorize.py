
import numpy as np
from math import log
from openfermion import (
    QubitOperator,
    get_sparse_operator
)
import networkx as nx
from csf_categorizer import categorize_csf_state


def factorize_state(basis_state, S_w, S_v, S_n):
    """
    Factorize a quantum state into tensor product components based on orbital sets.

    Input:
        basis_state: [determinants, indices, coefficients] where
            - determinants: list of occupation number arrays
            - indices: list of integer indices in 2^n_orb space
            - coefficients: array of coefficients for each determinant
        S_w: list of spatial orbital indices for W block (quantum treatment needed)
        S_v: list of spatial orbital indices for V block (affected by rotations)
        S_n: list of spatial orbital indices for N block (invariant)

    Return:
        dict mapping orbital tuples to state vectors:
        {(qubits_w...): psi_w, (qubits_v...): psi_v, (i,): psi_i, (j,): psi_j, ...}

    The keys are tuples of spin-orbital indices, values are normalized state vectors.
    """
    (_, k, l, a, b) = categorize_csf_state(basis_state)

    # Extract data from basis_state
    determinants = basis_state[0]
    indices = basis_state[1]
    coefficients = basis_state[2]

    # Determine number of spin-orbitals from the determinants
    n_spin_orbitals = len(determinants[0])
    n_spatial_orbitals = n_spin_orbitals // 2

    # Convert spatial orbital indices to spin-orbital indices
    # For each spatial orbital i, we have spin-orbitals 2*i (spin-up) and 2*i+1 (spin-down)
    spin_orb_W = sorted([2*i for i in S_w] + [2*i+1 for i in S_w])
    spin_orb_V = sorted([2*i for i in S_v] + [2*i+1 for i in S_v])
    spin_orb_N = sorted([2*i for i in S_n] + [2*i+1 for i in S_n])

    # All spin-orbitals that are part of factorization
    all_spin_orbs = spin_orb_W + spin_orb_V + spin_orb_N

    # Construct the full sparse state vector in 2^n_spin_orbitals space
    # Only populate the non-zero entries specified by indices and coefficients
    full_state = np.zeros(2**n_spin_orbitals, dtype=complex)
    for idx, coef in zip(indices, coefficients):
        full_state[idx] = coef

    # Determine how to factorize S_V based on the unique orbitals
    unique_orbitals = set(i for i in [k, l, a, b] if i is not None)

    factorization_dict = {}

    # Handle S_W block (if non-empty)
    if len(spin_orb_W) > 0:
        factorization_dict[tuple(spin_orb_W)] = extract_factor_from_state(
            full_state, tuple(spin_orb_W), n_spin_orbitals
        )

    # Handle S_V block with different factorization strategies
    if len(unique_orbitals) == 0:
        # S_V is empty - nothing to factorize
        pass
    elif len(unique_orbitals) == 2:
        # S_V cannot be factorized further - keep as one block
        if len(spin_orb_V) > 0:
            factorization_dict[tuple(spin_orb_V)] = extract_factor_from_state(
                full_state, tuple(spin_orb_V), n_spin_orbitals
            )
    elif len(unique_orbitals) == 4:
        # S_V can be factorized into individual spatial orbitals or pairs
        # For now, keep as one block (can be refined based on specific requirements)
        if len(spin_orb_V) > 0:
            factorization_dict[tuple(spin_orb_V)] = extract_factor_from_state(
                full_state, tuple(spin_orb_V), n_spin_orbitals
            )

    # Handle S_N block - factorize into individual spatial orbitals
    for spatial_idx in S_n:
        # Each spatial orbital i contributes spin-orbitals (2*i, 2*i+1)
        spin_pair = (2*spatial_idx, )
        factorization_dict[spin_pair] = extract_factor_from_state(
            full_state, spin_pair, n_spin_orbitals
        )
        factorization_dict[(2*spatial_idx + 1,)] = extract_factor_from_state(
            full_state, (2*spatial_idx + 1,), n_spin_orbitals
        )

    return factorization_dict


def extract_factor_from_state(full_state, qubit_indices, n_total_qubits):
    """
    Extract a tensor factor from a full quantum state.

    This function traces out all qubits except those in qubit_indices,
    assuming the state factorizes (i.e., is a product state).

    Args:
        full_state: Full state vector in 2^n_total_qubits dimensional space
        qubit_indices: tuple of qubit indices to extract
        n_total_qubits: total number of qubits

    Returns:
        Normalized state vector for the specified qubits (dimension 2^len(qubit_indices))
    """
    n_factor_qubits = len(qubit_indices)
    factor_state = np.zeros(2**n_factor_qubits, dtype=complex)

    # Find one non-zero entry in the full state to use as reference
    nonzero_idx = np.nonzero(full_state)[0]
    if len(nonzero_idx) == 0:
        # State is zero - return zero vector
        return factor_state

    # Use first non-zero entry as reference
    ref_global_idx = nonzero_idx[0]

    # Convert global index to binary representation
    ref_bits = [(ref_global_idx >> i) & 1 for i in range(n_total_qubits)]

    # Extract values for qubits NOT in qubit_indices (complementary qubits)
    comp_qubits = [i for i in range(n_total_qubits) if i not in qubit_indices]
    comp_bits = [ref_bits[i] for i in comp_qubits]

    # Iterate through all possible configurations of the factor qubits
    for factor_idx in range(2**n_factor_qubits):
        # Convert factor index to bits
        factor_bits = [(factor_idx >> i) & 1 for i in range(n_factor_qubits)]

        # Construct full index by combining factor bits and complementary bits
        full_bits = [0] * n_total_qubits
        for i, q in enumerate(qubit_indices):
            full_bits[q] = factor_bits[i]
        for i, q in enumerate(comp_qubits):
            full_bits[q] = comp_bits[i]

        # Convert bits to index
        global_idx = sum(bit << i for i, bit in enumerate(full_bits))

        # Extract amplitude
        factor_state[factor_idx] = full_state[global_idx]

    # Normalize
    norm = np.linalg.norm(factor_state)
    if norm > 1e-12:
        factor_state /= norm

    return factor_state


def get_indices_mapping_2_wvn(basis_state, mp2_amplitude, Norb):
    """
    Args:
        basis_state: list of occupied spin-orbital indices in the CSF basis state (list_list_refCSF[i][j])
        mp2_amplitude: list of MP2 amplitude data [[[i, a]], amplitude_value (list_list_Uext_mp2_ampld[i])
        Norb: number of spatial orbitals
    Return: {index: 'W' or 'V' or 'N'}

    """
    S_W, S_V, S_N = [], [], []
    
    if not mp2_amplitude == []:
        print(f'MP2 amplitude data provided: {mp2_amplitude}')
        
        # Extract the spatial indices that have to be quantumly treated
        for amplitude_data in mp2_amplitude:
            # amplitude_data format: [[[0, 5]], amplitude_value]
            # Extract the indices from the first element
            indices = amplitude_data[0][0]  # Gets [0, 5] from [[[0, 5]], amplitude_value]
            print(f'Indices from MP2 amplitude data: {indices}')
            S_W.extend(indices)  # Add both indices to S_W
        S_W = list(set(S_W))
    
    # Extract indices that are affected by Vu rotations
    (_, k, l, a, b) = categorize_csf_state(basis_state)
    print(f'Categorized CSF state indices: k={k}, l={l}, a={a}, b={b}')
    # Convert spin-orbitals to spatial orbitals (integer division by 2), skip None values
    spatial_orbitals = [idx for idx in [k, l, a, b] if idx is not None]
    S_V = list(set([idx for idx in spatial_orbitals]))
    # Get the invariant indices under Vu and Wu rotations
    S_N = [k for k in range(Norb) if k not in S_W and k not in S_V]
    return {index: 'W' for index in S_W} | {index: 'V' for index in S_V} | {index: 'N' for index in S_N}

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

