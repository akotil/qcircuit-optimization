import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SdgGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
import warnings
import optimizations
warnings.filterwarnings("ignore", category=DeprecationWarning)


H: np.array = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
P: np.array = np.array([[np.exp(-1j * np.pi / 4), 0], [0, np.exp(1j * np.pi / 4)]])
P_dagger: np.array = np.array([[np.exp(1j * np.pi / 4), 0], [0, np.exp(-1j * np.pi / 4)]])
S = np.array([[1, 0], [0, 1j]])
S_dagger = np.array([[1, 0], [0, -1j]])
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
T_dagger = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

u1 = H @ P @ H
u2 = P_dagger @ H @ P_dagger
#print(u1 / u2)

#print("---------")

u3 = H @ T @ H
u4 = T_dagger @ H @ T_dagger
#print(u3 / u4)

Rz = lambda x : np.array([[np.exp(-1j*x/2), 0], [0, np.exp(1j*x/2)]])
IRz1 = np.kron(np.eye(2), H)
IRz2 = np.kron(np.eye(2), Rz(-np.pi/2))

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

print("They commute: ", np.allclose(CNOT @ IRz1 @ CNOT @ IRz2, IRz2 @ CNOT @ IRz1 @ CNOT))

class TestEquivalences(unittest.TestCase):

    # H ⊗ H - CNOT - H ⊗ H
    def test_cnot_transformation(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0,1)
        qc.h(0)
        qc.h(1)

        qc.rz(np.pi, 0)
        qc.rz(np.pi, 1)

        qc.h(0)
        qc.h(1)
        qc.cx(0,1)
        qc.h(0)
        qc.h(1)

        return qc

    # S gate transformation
    def test_p_transformation_1(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        return qc

    # S dagger gate transformation
    def test_p_transformation_2(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        sgd = SdgGate()
        qc.append(sgd, [0])
        qc.h(0)
        return qc

    # H - P - CNOT - P_dagger - H = P_dagger - CNOT - P
    def test_p_transformation_3(self):
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.s(1)
        qc.cx(0, 1)
        sdg = SdgGate()
        qc.append(sdg, [1])
        qc.h(1)
        return qc

    # H - P_dagger - CNOT - P - H = P - CNOT - P_dagger
    def test_p_transformation_4(self):
        qc = QuantumCircuit(2)
        qc.h(1)
        sdg = SdgGate()
        qc.append(sdg, [1])
        qc.cx(0, 1)
        qc.s(1)
        qc.h(1)
        return qc

    def test_rz_commutation_1(self):
        qc = QuantumCircuit(3)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.cnot(2,0)
        qc.h(0)
        qc.x(0)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.cnot(2,0)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.cnot(0,2)
        qc.rz(np.pi, 0)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.cx(1, 2)
        qc_ref.cx(0, 1)

        # Apply the optimization procedure
        reduction = optimizations.RzReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        assert qc_ref == qc_optimized

        return qc

    def test_rz_commutation_2(self):
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 1)
        qc.cnot(0,1)
        qc.rz(-np.pi, 1)
        qc.cnot(0,1)
        qc.rz(np.pi, 1)
        return qc


    def test_cx_commutation_1(self):
        qc = QuantumCircuit(3)
        qc.cx(0,2)
        qc.cx(1,2)
        qc.cx(0,1)
        qc.cx(0,2)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.cx(1,2)
        qc_ref.cx(0,1)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        assert qc_ref == qc_optimized


    def test_cx_commutation_2(self):
        qc = QuantumCircuit(3)
        qc.cx(0,2)
        qc.cx(1,2)
        qc.cx(0,1)
        qc.h(2)
        qc.cx(0,2)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        # The circuit should not change
        assert qc== qc_optimized


    def test_cx_commutation_3(self):
        qc = QuantumCircuit(3)
        qc.cx(0,1)
        qc.h(1)
        qc.cx(1,2)
        qc.h(1)
        qc.cx(0,2)
        qc.cx(0,1)
        qc.h(0)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.h(1)
        qc_ref.cx(1,2)
        qc_ref.h(1)
        qc_ref.cx(0,2)
        qc_ref.h(0)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        reduction.animate()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        assert qc_ref == qc_optimized