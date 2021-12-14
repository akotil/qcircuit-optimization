import unittest
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SdgGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

import optimizations

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestEquivalences(unittest.TestCase):

    # H ⊗ H - CNOT - H ⊗ H
    def test_cnot_transformation(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)

        qc.rz(np.pi, 0)
        qc.rz(np.pi, 1)

        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
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
        qc.cnot(2, 0)
        qc.h(0)
        qc.x(0)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.cnot(2, 0)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.cnot(0, 2)
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
        qc.cnot(0, 1)
        qc.rz(-np.pi, 1)
        qc.cnot(0, 1)
        qc.rz(np.pi, 1)
        return qc

    def test_cx_commutation_1(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.cx(1, 2)
        qc_ref.cx(0, 1)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        assert qc_ref == qc_optimized

    def test_cx_commutation_2(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(0, 2)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        # The circuit should not change
        assert qc == qc_optimized

    def test_cx_commutation_3(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(1, 2)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        qc.h(0)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.h(1)
        qc_ref.cx(1, 2)
        qc_ref.h(1)
        qc_ref.cx(0, 2)
        qc_ref.h(0)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        reduction.animate()
        qc_optimized = dag_to_circuit(dag)
        print(qc_optimized.draw())

        assert qc_ref == qc_optimized
