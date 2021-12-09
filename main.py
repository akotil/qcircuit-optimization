from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector
from optimizations import HGateReduction, CxReduction, RzReduction

import converters

# Convert the input to a quantum circuit
parser = converters.Parser('./inputs/vbe_adder_3_before')
netlist = parser.qc_to_netlist()
input_qc = parser.netlist_to_qiskit_circuit(netlist)
input_qc.draw(output='mpl', filename="./outputs/test_old")

parser_ref = converters.Parser('./inputs/vbe_adder_3_after_light')
ref_qc = parser_ref.netlist_to_qiskit_circuit(parser_ref.qc_to_netlist())
ref_qc.draw(output='mpl', filename="./outputs/test_ref")
assert (Statevector.from_instruction(input_qc).equiv(Statevector.from_instruction(ref_qc)))
dag = circuit_to_dag(input_qc)

# Preserving the order presented in the paper
routine_dic = {1: "HGateReduction", 2: "RzReduction", 3: "CxReduction"}
# todo: add 4th routine
light_optimization = [1, 3, 2, 3, 1, 2, 3, 2]

# Apply the optimization procedure
for i in range(2):
    for routine in light_optimization:
        klass = globals()[routine_dic[routine]]
        reduction = klass(dag)
        dag = reduction.apply()

optimized_qc = dag_to_circuit(dag)

# TODO: Look at the QASM outputs
assert Statevector.from_instruction(ref_qc).equiv(Statevector.from_instruction(optimized_qc))
print(input_qc.draw())
# Draw the quantum circuit after optimization
qc = dag_to_circuit(dag)
qc.draw(output='mpl', filename="./outputs/test_new")
