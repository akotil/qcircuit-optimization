# qcircuit-optimization

Python/Qiskit implementation of some of the quantum circuit optimization procedures described in https://doi.org/10.1038/s41534-018-0072-4.

# Prerequisites

Run 'pip install -r requirements.txt' to install all the requirements. The installation in a virtual environment is recommended.

# Usage

`optimizations.py` contains three optimization procedures from the paper. `HGateReduction` corresponds to the first optimization subroutine 'Hadamard gate reduction'. `RzReduction` corresponds to the second subroutine 'Single qubit gate cancellation'. Finally, `CxReduction` corresponds to the third subroutine 'Two-qubit gate cancellation'.

The reduction classes expect a `DAGCircuit` object ('directed acylic graph' circuit implemented by Qiskit, see https://qiskit.org/documentation/stubs/qiskit.dagcircuit.DAGCircuit.html) by initialization. A `DAGCircuit` object may be obtained by the converter function `circuit_to_dag` provided by Qiskit. Alternatively, one can also use the ASCII format of the Quipper language (https://www.mathstat.dal.ca/~selinger/quipper/) corresponding to a quantum circuit within a file. This file may be converted by the `Parser` contained in `converters.py`. The Quipper format is first converted into a list containing the gate information of the quantum circuit (gate name:str, qubit:int, control qubits:list, inverted:bool). Then, the list is converted into a `QuantumCircuit` object (see https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html):

```python
parser = converters.Parser('./inputs/vbe_adder_3_before')
netlist = parser.qc_to_netlist()
input_qc = parser.netlist_to_qiskit_circuit(netlist)
```

As an example, Hadamard Gate reduction is accomplished as follows:

```python
qc = QuantumCircuit(3)
qc.cx(0, 2)
qc.h(1)

dag = circuit_to_dag(input_qc)
reduction = HGateReduction(dag)
dag = reduction.apply()
```

The optimized circuit may again be obtained via the Qiskit converter `dag_to_circuit`.

In order to obtain information about the reduced gate counts, one can use `reduction.report()` after the reduction has been applied.

