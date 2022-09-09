import numpy as np
from sympy import symbols
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    par = [0.5]
    
    def f(*pars):
        par_dict = dict(zip(sym, pars))
        circ_par = circ.copy()
        circ_par.symbol_substitution(par_dict)
        v = np.real(get_operator_expectation_value(circ_par, op, backend, n_shots=1000, partition_strat=PauliPartitionStrat.CommutingSets))
        print(pars, v)
        return v

    #res = minimize(f, par, method="BFGS")
    #print(res)
    pars = [-2 + 0.05*i for i in range(80)]
    vals = [f(p) for p in pars]
    grads = [get_gradient(op, circ, [p], sym, backend, shots=1000) for  p in pars]

    import pandas as pd
    df = pd.DataFrame()
    df["par"] = pars
    df["val"] = vals
    df["grad"] = grads
    df.to_csv("experiment1_1000shots.csv")
    plt.plot(pars, vals)
    plt.plot(pars, grads)
    plt.show()

    if False:
        tol = 1e-2; step = 0.01; max_n = 100; n=0
        best_par = list(par)
        best_val = f(par)
        last_best_val = 10**9
        pars, vals = [best_par], [best_val]

        while np.abs(best_val - last_best_val)/last_best_val > tol and n < max_n:
            grad = get_gradient(op, circ, pars[-1], sym, backend)
            print(pars[-1], grad, vals[-1])
            new_par = list(np.array(pars[-1]) - step*np.array(grad))
            new_val = f(new_par)
            pars += [new_par]; vals += [new_val]
            if new_val < best_val:
                last_best_val = best_val
                best_val = new_val
                best_par = new_par
            n += 1
        
        print("Minimum: ", best_par, best_val)

        plt.plot(vals)
        plt.show()


    #circ, syms, parameters = get_randomized_circuit(3, 2)
    #print(circ)
    #grad = get_gradient(op, circ, parameters, syms, backend)

    #print(grad)