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



def build_test_circuit(par):
    circ = Circuit(1)
    circ.H(0)
    circ.Ry(angle=par[0], qubit=0)
    #circ.Rx(angle=par[1], qubit=0)
    circ.measure_all()
    return circ

def get_one_derivative(op, i, circuit_f, par, backend):
    """ arguments:
    op - operator
    i - index of parameter to differentiate
    circuit_f - function that creates the curcuit with parameters par
    par - list of parameter values
    backend - backend to sue for gradient computation
    
    returns:
    (float) the i-th element of the gradient vector
    """
    circ_plus = circuit_f(par[:i] + [par[i]+np.pi/2] + [par[i+1:]])
    circ_minus = circuit_f(par[:i] + [par[i] - np.pi/2] + [par[i+1:]])

    exp_val_left =  get_operator_expectation_value(circ_plus, op, backend, n_shots=100, partition_strat=PauliPartitionStrat.CommutingSets)
    exp_val_right =  get_operator_expectation_value(circ_minus, op, backend, n_shots=100, partition_strat=PauliPartitionStrat.CommutingSets)
    
    return 0.5*(np.real(exp_val_left) - np.real(exp_val_right))

def get_gradient(op, circuit_f, par, backend):
    """ arguments:
    op - operator
    circuit_f - function that creates the curcuit with parameters par
    par - list of parameter values
    backend - backend to sue for gradient computation

    returns:
    (list of floats): the gradient vector
    """
    grad = []
    for i in range(len(par)):
        dTheta = get_one_derivative(op, i, circuit_f, par, backend)
        grad += [dTheta]
    
    return grad

if __name__ == "__main__":
    backend = AerBackend()

    z = QubitPauliString({
            Qubit(0) : Pauli.Z})

    op = QubitPauliOperator({
            z : 1})

    par = [0.7, 0.4]
    circ = build_test_circuit(par)
    print(circ)
    grad = get_gradient(op, build_test_circuit, par, backend)

    print(grad)