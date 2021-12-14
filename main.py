from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector
from optimizations import HGateReduction, CxReduction, RzReduction

import converters

# Convert the input to a quantum circuit
parser = converters.Parser('./inputs/vbe_adder_3_before')
netlist = parser.qc_to_netlist()
input_qc = parser.netlist_to_qiskit_circuit(netlist)
input_qc.draw(output='mpl', filename="./outputs/test_input")
dag = circuit_to_dag(input_qc)

# Preserving the order of 'Light Optimization' as presented in the paper (without the 4th optimization procedure)
routine_dic = {1: "HGateReduction", 2: "RzReduction", 3: "CxReduction"}
light_optimization = [1, 3, 2, 3, 1, 2, 3, 2]

# Apply the optimization procedure
for i in range(2):
    for routine in light_optimization:
        klass = globals()[routine_dic[routine]]
        reduction = klass(dag)
        dag = reduction.apply()
        reduction.report()

optimized_qc = dag_to_circuit(dag)
assert Statevector.from_instruction(input_qc).equiv(Statevector.from_instruction(optimized_qc))

# Draw the quantum circuit after optimization
qc = dag_to_circuit(dag)
qc.draw(output='mpl', filename="./outputs/test_optimized_input")
