import numpy as np

from pytket import Circuit, Qubit
from pytket.circuit import Pauli, PauliExpBox
from pytket.circuit.display import render_circuit_jupyter 
from pytket.utils import get_operator_expectation_value

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.partition import PauliPartitionStrat

from sympy import symbols



def build_test_circuit():
    a = symbols("a")
    circ = Circuit(1)
    #circ.H(0)
    circ.Ry(angle=a, qubit=0)
    #circ.Ry(angle=b, qubit=1)
    #circ.measure_all()
    return circ


def get_one_derivative(i, op, circuit, par, sym, backend):
    """ arguments:
    op - operator
    i - index of parameter to differentiate
    circuit_f - function that creates the curcuit with parameters par
    par - list of parameter values
    sym - list of symbols used in the symbolic circuit
    backend - backend to sue for gradient computation
    
    returns:
    (float) the i-th element of the gradient vector
    """
    # make copies of the ciruit
    circ_plus = circuit.copy()
    circ_minus = circuit.copy()
    # pair up symbols and parameter values and form a dictionary
    par_dict_plus = dict(zip(sym, par))
    par_dict_minus = dict(zip(sym, par))
    print(par_dict_minus)
    # perform the parameter shift on the i-th parameter value
    par_dict_plus[sym[i]] = par[i] + np.pi/2
    par_dict_minus[sym[i]] = par[i] - np.pi/2
    # subsitute the symbols in the dict with their parameter value
    circ_plus.symbol_substitution(par_dict_plus)
    circ_minus.symbol_substitution(par_dict_minus)
    # estimate the expectation values
    exp_val_left =  get_operator_expectation_value(circ_plus, op, backend, n_shots=10, partition_strat=PauliPartitionStrat.CommutingSets)
    exp_val_right =  get_operator_expectation_value(circ_minus, op, backend, n_shots=10, partition_strat=PauliPartitionStrat.CommutingSets)
    # compute the derivative according to the parameter shift rule for Pauli matrices
    return 0.5*(np.real(exp_val_left) - np.real(exp_val_right))

def get_gradient(op, circuit, par, symbols, backend):
    """ arguments:
    op - operator
    circuit_f - function that creates the curcuit with parameters par
    par - list of parameter values
    sym - list of symbols used in the symbolic circuit
    backend - backend to sue for gradient computation

    returns:
    (list of floats): the gradient vector
    """
    grad = []
    for i in range(len(par)):
        dTheta = get_one_derivative(i, op, circuit, par, symbols, backend)
        grad += [dTheta]
    
    return grad

if __name__ == "__main__":
    backend = AerBackend()

    z = QubitPauliString({
            Qubit(0) : Pauli.Z})

    op = QubitPauliOperator({
            z : 1})

    #par = [0.7, 0.4]
    circ = build_test_circuit()
    print(circ)
    grad = get_gradient(op, circ, [0.49], [symbols("a")], backend)

    print(grad)