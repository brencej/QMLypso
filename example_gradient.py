import numpy as np
from sympy import symbols

from pytket import Circuit, Qubit
from pytket.circuit import Pauli
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString

from gradient import get_gradient

def get_test_circuit():
    a = symbols("a")
    circ = Circuit(1)
    #circ.H(0)
    circ.Ry(angle=a, qubit=0)
    #circ.Ry(angle=b, qubit=1)
    #circ.measure_all()
    return circ

def get_randomized_circuit(n_par, n_q):
    # circuit with random no of parameters and random number of qubits (but fixed gates) for testing stuff, symbolic
    syms = [symbols('syms%d' % i) for i in range(n_par)]
    circ = Circuit(n_q)
            
    for q in range(n_q): 
        circ.H(q)
    gate_type = [circ.Rx, circ.Ry, circ.Rz]
    params = []
    for i in range(n_par):
        params += [np.random.random()*2]
        np.random.choice(gate_type)(angle = params[-1], qubit = np.random.randint(0, n_q))
    
    return circ, syms, params


if __name__ == "__main__":
    backend = AerBackend()

    z = QubitPauliString({
            Qubit(0) : Pauli.Z})

    op = QubitPauliOperator({
            z : 1})

    circ, syms, parameters = get_randomized_circuit(3, 2)
    print(circ)
    grad = get_gradient(op, circ, parameters, syms, backend)

    print(grad)