import numpy as np
from openfermion import get_sparse_operator

def expectation(Op, State):
    return (State @ Op @ State.T)[0,0]

def matrix_element(Op, Bra, Ket):
    return (Bra @ Op @ Ket.T)[0,0]

def variance_of_operator(Op, State):
    """
    computes the variance of a Hermitian operator <psi|H^2|psi> - <psi|H|psi>^2
    """
    first  = (State @ Op) @ (Op @ State.T)
    second = (State @ Op @ State.T) ** 2
    return first - second

def variance_of_general_operator(Op, State):
    """
    computes the variance of a general non-Hermitian operator<psi|H^t H|psi> - <psi|H^t|psi><psi|H|psi>

    note that qubit Hamiltonians with complex coefficients are not Hermitian. But a QWC or FC Hamiltonian can be measured independent of
    what the coefficients are
    """
    first  = (State @ Op.conjugate().transpose()) @ (Op @ State.T)
    second = (State @ Op @ State.T)
    third  = (State @ Op.conjugate().transpose() @ State.T) 
    return first - (second * third)

def variance_of_decomp(decomp, State, N, general=False):
    if not general:
        var_list = [variance_of_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    else:
        var_list = [variance_of_general_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    var_list = np.array([x.toarray()[0,0] for x in var_list])
    return np.sum((var_list)**(1/2))**2

