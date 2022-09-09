import numpy as np

from pytket import Circuit, Qubit
from pytket.utils import get_operator_expectation_value
from pytket.partition import PauliPartitionStrat


def get_partial_derivative(i, op, circuit, par, sym, backend, shots=100):
    """ Computes the partial derivative in respect to parameter i of the expectation value 
    of operator op, using the parameter shift rule. Currently supports only rotation gates.
    Arguments:
    i (int): index of parameter to differentiate
    op (Operator): operator
    circuit (Circuit): symbolic parametrized circuit
    par (list of floats): list of parameter values
    sym (list of SymPy symbols): list of symbols used in the symbolic circuit
    backend (Backend): backend to use for gradient computation
    
    Returns:
    (float) the i-th element of the gradient vector
    """
    # make copies of the ciruit
    circ_plus = circuit.copy()
    circ_minus = circuit.copy()
    # pair up symbols and parameter values and form a dictionary
    par_dict_plus = dict(zip(sym, par))
    par_dict_minus = dict(zip(sym, par))
    # perform the parameter shift on the i-th parameter value
    par_dict_plus[sym[i]] = par[i] + np.pi/2
    par_dict_minus[sym[i]] = par[i] - np.pi/2
    # subsitute the symbols in the dict with their parameter value
    circ_plus.symbol_substitution(par_dict_plus)
    circ_minus.symbol_substitution(par_dict_minus)
    # estimate the expectation values
    exp_val_left =  get_operator_expectation_value(circ_plus, op, backend, n_shots=shots, partition_strat=PauliPartitionStrat.CommutingSets)
    exp_val_right =  get_operator_expectation_value(circ_minus, op, backend, n_shots=shots, partition_strat=PauliPartitionStrat.CommutingSets)
    # compute the derivative according to the parameter shift rule for Pauli matrices
    return 0.5*(np.real(exp_val_left) - np.real(exp_val_right))

def get_gradient(op, circuit, par, symbols, backend, shots=100):
    """ Computes the gradient of the expectation value of operator op, 
    using the parameter shift rule. Currently supports only rotation gates.
    Arguments:
    op (Operator): operator
    circuit (Circuit): symbolic parametrized circuit
    par (list of floats): list of parameter values
    sym (list of SymPy symbols): list of symbols used in the symbolic circuit
    backend (Backend): backend to use for gradient computation
    
    Returns:
    (list of floats): the gradient vector
    """
    grad = []
    for i in range(len(par)):
        dTheta = get_partial_derivative(i, op, circuit, par, symbols, backend, shots=shots)
        grad += [dTheta]
    
    return grad

