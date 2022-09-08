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
    circ.Ry(0.7, 0)
    circ.measure_all()
    return circ

def find_one_derivative(i, circuit_f, par, backend):
    """ inputs:
    i - index of parameter to differentiate
    circuit_f - function that creates the curcuit with parameters par
    par - list of parameter values
    backend - backend to sue for gradient computation
    """
    circ_plus = circuit_f(par[:i] + [par[i]+np.pi/2] + [par[i+1:]])
    circ_minus = circuit_f(par[:i] + [par[i] - np.pi/2] + [par[i+1:]])

    #...
    d_theta = None
    return d_theta

if __name__ == "__main__":
    backend = AerBackend()

    z = QubitPauliString({
            Qubit(0) : Pauli.Z})

    op = QubitPauliOperator({
            z : 1})

    exp_val = get_operator_expectation_value(
            circ,
            op,
            backend,
            n_shots=10000,
            partition_strat=PauliPartitionStrat.CommutingSets)