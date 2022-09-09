import numpy as np
from sympy import symbols

from pytket import Circuit, Qubit
from pytket.circuit import Pauli
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import get_operator_expectation_value
from pytket.partition import PauliPartitionStrat


from gradient import get_gradient

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

def get_test_circuit():
    a = symbols("a")
    circ = Circuit(1)
    #circ.H(0)
    circ.Ry(angle=a, qubit=0)
    #circ.Ry(angle=b, qubit=1)
    #circ.measure_all()
    return circ, a

if __name__ == "__main__":
    backend = AerBackend()

    z = QubitPauliString({
            Qubit(0) : Pauli.Z})

    op = QubitPauliOperator({
            z : 1})

    circ, a = get_test_circuit()
    sym = [a]
    par = [1/5]

    def f(par):
        par_dict = dict(zip(sym, par))
        circ_par = circ.copy()
        circ_par.symbol_substitution(par_dict)
        return get_operator_expectation_value(circ_par, op, backend, n_shots=10, partition_strat=PauliPartitionStrat.CommutingSets)

    tol = 1e-3; step = 0.05
    best_par = list(par)
    best_val = f(par)
    pars, vals = [best_par], [best_val]

    
    print("f(0) = ", best_val)
    print("grad(0) = ", get_gradient(op, circ, par, sym, backend))

    #circ, syms, parameters = get_randomized_circuit(3, 2)
    #print(circ)
    #grad = get_gradient(op, circ, parameters, syms, backend)

    #print(grad)