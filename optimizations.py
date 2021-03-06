import numpy as np
from qiskit.circuit.library import HGate, RZGate
from qiskit.dagcircuit import DAGCircuit, DAGNode


class Reduction:
    """
    Base class for the following reduction types: HGateReduction, RzReduction and CxReduction.

    The reduction procedures are all obtained from the following paper: https://doi.org/10.1038/s41534-018-0072-4
    Only the first three reductions are implemented.
    """

    def __init__(self, dag: DAGCircuit):
        """
        Initializes the reduction procedure.

        :param dag: The circuit to be optimized
        """
        self.dag = dag
        self.initial_counts = dag.count_ops()

    def report(self):
        """
        Reports the gate count changes in the circuit (if there are any) after the reduction has been applied.
        """

        optimized_counts = self.dag.count_ops()
        for gate in self.initial_counts:
            if gate in optimized_counts:
                if self.initial_counts[gate] > optimized_counts[gate]:
                    print("Reduced the number of %s gates by %d (Before reduction: %d, After reduction: %d)\n" %
                          (gate, self.initial_counts[gate] - optimized_counts[gate], self.initial_counts[gate],
                           optimized_counts[gate]))
            else:
                print("Successfully eliminated all %d %s gates." % (self.initial_counts[gate], gate))

    def apply(self):
        raise NotImplementedError("Implement the apply method")


class HGateReduction(Reduction):
    '''
    Returns the given DAG with reduced number of Hadamard Gates if any reduction is possible.
    '''

    def __init__(self, dag: DAGCircuit):
        super().__init__(dag)

    def apply(self):
        '''
        Iterate through the nodes in the graph to look for the following gate sequences:

        - H - P - H
        - H - P† - H
        - H ⊗ H - CNOT - H ⊗ H
        - H - P - CNOT - P† - H
        - H - P† - CNOT - P - H

        If a sequence is found, it is replaced by its equivalent sequence which has less Hadamard Gates.
        '''

        # First, remove all redundant Hadamard Gates (i.e. cancel adjacent Hadamard Gates)
        for edge in self.dag.edges():
            source, dest, _ = edge
            if source.name == "h" and dest.name == "h":
                self.dag.remove_op_node(source)
                self.dag.remove_op_node(dest)

        # Phase Gate transformations:
        # H - P - H = P† - H - P†
        # H - P† - H = P - H - P
        for wire in self.dag.wires:
            wire_nodes = self.dag.nodes_on_wire(wire, True)
            sequence = []
            node: DAGNode

            # Go through every node in the wire to find a valid phase gate sequence
            for node in wire_nodes:
                if node.name == "h" and sequence == []:
                    sequence.append(node)
                elif node.name == "h" and len(sequence) == 2:
                    s_gate = sequence[1].op.inverse()
                    # Substitute the found sequence by the equivalent sequence
                    # which has reduced number of Hadamard Gates
                    self.dag.substitute_node(sequence[0], sequence[1].op.inverse(), inplace=True)
                    self.dag.substitute_node(sequence[1], HGate(), inplace=True)
                    self.dag.substitute_node(node, s_gate, inplace=True)
                elif self._is_phase_gate(node) and len(sequence) == 1:
                    sequence.append(node)
                else:
                    sequence = []

        # H ⊗ H - CNOT - H ⊗ H transformations
        for node in self.dag.nodes():
            if node.name == "cx":
                predecessors = list(self.dag.predecessors(node))
                successors = list(self.dag.successors(node))

                # For every CNOT Gate, check if all the incoming and outgoing edges of the corresponding operation node
                # are connected to nodes which correspond to Hadamard Gates
                self.dag.edges(nodes=[node])
                found = False
                if len(predecessors) == 2 and len(predecessors) == 2:
                    if all(pred.name == "h" for pred in predecessors) and all(succ.name == "h" for succ in successors):
                        found = True

                # When the sequence is found, the Hadamard Gates are removed and the CNOT Gate is flipped
                if found:
                    for pred in predecessors:
                        self.dag.remove_op_node(pred)
                    for suc in successors:
                        self.dag.remove_op_node(suc)
                    node.qargs.reverse()

        # H - P - CNOT - P† - H = P† - CNOT - P
        # H - P† - CNOT - P - H = P - CNOT - P†
        for wire_idx, wire in enumerate(self.dag.wires):
            wire_nodes = self.dag.nodes_on_wire(wire, True)
            sequence = []
            node: DAGNode
            # Go through every node in the wire to find a valid phase gate sequence
            for node in wire_nodes:
                if node.name == "h" and sequence == []:
                    sequence.append(node)
                elif node.name == "h" and len(sequence) == 4:
                    # The sequence is found -> remove the Hadamard Gates and invert the Phase Gates inplace
                    self.dag.remove_op_node(sequence[0])
                    self.dag.remove_op_node(node)
                    self.dag.substitute_node(sequence[1], sequence[1].op.inverse(), inplace=True)
                    self.dag.substitute_node(sequence[3], sequence[3].op.inverse(), inplace=True)
                elif self._is_phase_gate(node) and len(sequence) == 1:
                    sequence.append(node)
                elif node.name == "cx" and len(sequence) == 2:
                    # The target of the CNOT node from the sequence must be on the same wire as other nodes
                    target = node.qargs[1].index
                    if target == wire_idx:
                        sequence.append(node)
                    else:
                        sequence = []
                elif len(sequence) == 3 and node.op == sequence[1].op.inverse():
                    sequence.append(node)
                else:
                    sequence = []
        return self.dag

    def _is_phase_gate(self, node: DAGNode):
        """
        A node represents a phase gate if
            - the gate is an S gate (or S†) or
            - the gate is Rz(pi/2)
        :param node: the node to be checked
        :return: True if node represents a phase gate
        """
        if node.name == "s" or node.name == "sdg":
            return True
        elif node.name == "rz" and abs(node.op.params[0]) == np.pi / 2:
            return True
        return False


class RzReduction(Reduction):
    '''
    Returns the given DAG with reduced number of Rz Gates if any reduction is possible.
    '''

    def __init__(self, dag: DAGCircuit):
        super().__init__(dag)
        self.deleted_nodes = []
        self.merged_nodes = {}

    def apply(self):
        """
        The reduction can be applied if there is at least one rotation gate which cancels or merges with
        another rotation gate. The cancellation/merge may be applied if the two rotation gates are adjacent up to
        some commutation rules.

        The commutation rules are:
        - Rz - H - CNOT - H = H - CNOT - H - Rz
        - Rz - CNOT - Rz' - CNOT = CNOT - Rz' - CNOT - Rz
        - Rz - CNOT = CNOT - Rz
        """

        for wire in self.dag.wires:
            wire_nodes = self.dag.nodes_on_wire(wire, True)
            node: DAGNode
            cursor = 0
            for i, node in enumerate(wire_nodes):
                # The cursor indicates the index of the last merged/cancelled rotation gate
                # If no cancel/merge was possible, the cursor simply moves forward on the wire by one node
                if i < cursor:
                    continue
                if node.name == "rz":
                    cursor = self._search(i, wire, node)

        for node in self.deleted_nodes:
            self.dag.remove_op_node(node)

        for key in self.merged_nodes:
            self.dag.substitute_node(key, RZGate(self.merged_nodes[key]), inplace=True)

        return self.dag

    def _search(self, rz_index: int, wire, node: DAGNode) -> int:
        """
        Searches for and returns the index of a node which commutes with the given node corresponding to a RZ operation.
        If a commutation is not possible, the index of the next node on the wire is returned.

        :param rz_index: The index of the node to be commuted
        :param wire: The wire the node is attached to
        :param node: The the node to be commuted
        """

        wire_nodes = list(self.dag.nodes_on_wire(wire, True))
        if rz_index < len(wire_nodes):
            wire_nodes = wire_nodes[rz_index + 1:]
            i = 0
            while i < len(wire_nodes):
                # Look for three nodes ahead to check whether one of the 4 gate commutation rules apply
                if i + 3 < len(wire_nodes):
                    lookahead_nodes = wire_nodes[i:i + 3]
                    if self._commutes(lookahead_nodes, wire.index):
                        i += 3
                        continue
                # Check for a commutation with a CNOT Gate
                if i + 1 < len(wire_nodes) and wire_nodes[i].name == "cx":
                    if wire_nodes[i].qargs[0].index == wire.index:
                        i += 1
                        continue
                # If a further commutation is not possible but the current node is another rotation gate,
                # we cancel/merge
                if wire_nodes[i].name == "rz":
                    angle = wire_nodes[i].op.params[0]
                    self.deleted_nodes.append(node)
                    self.merged_nodes[wire_nodes[i]] = angle + node.op.params[0]
                    return rz_index + i + 2
                # Commutation is not possible, the search will continue on the next node on the wire
                return rz_index + 1
        # Commutation is not possible, the search will continue on the next node on the wire
        return rz_index + 1

    def _commutes(self, lookahead_nodes, wire_idx) -> bool:
        """
        Returns True if the given three nodes correspond to one of the two commutation sequences.

        :param lookahead_nodes: The nodes to be commuted
        :param wire_idx: The wire index the lookahead_nodes are on
        """
        names = [node.name for node in lookahead_nodes]
        if names == ["h", "cx", "h"]:
            cx = lookahead_nodes[1]
            # The target of the CNOT Gate must be on the same wire as other commuting nodes
            if cx.qargs[1].index == wire_idx:
                return True
        elif names == ["cx", "rz", "cx"]:
            cx_nodes = [lookahead_nodes[0], lookahead_nodes[2]]
            # The targets of the CNOT Gates must both be on the same wire as other commuting nodes
            if all(cx.qargs[1].index == wire_idx for cx in cx_nodes):
                return True
        return False


class CxReduction(Reduction):
    '''
    Returns the given DAG with reduced number of Cx Gates if any reduction is possible.
    '''

    def __init__(self, dag: DAGCircuit):
        super().__init__(dag)

    def apply(self):
        """
        The reduction can be applied if there is at least one CNOT Gate which cancels with another CNOT Gate.
        The cancellation may be applied if the two CNOT Gates are adjacent up to some commutation rules.

        The commutation rules are:
        - Two adjacent CNOT Gates with the same target qubit commute
        - Two adjacent CNOT Gates with the same control qubit commute
        - CNOT (target) - H - CNOT (control) - H = H - CNOT (control) - H - CNOT (target)
        """

        for wire_idx, wire in enumerate(self.dag.wires):
            wire_nodes = self.dag.nodes_on_wire(wire, True)
            node: DAGNode
            commuted_nodes = []
            for node_idx, node in enumerate(wire_nodes):
                if node.name == "cx" and node not in commuted_nodes:
                    control = node.qargs[0].index
                    target = node.qargs[1].index
                    if control == wire_idx:
                        other_wire = target
                        type = "control"
                        other_type = "target"
                    else:
                        other_wire = control
                        type = "target"
                        other_type = "control"
                    if abs(control - target) == 2 or abs(control - target) == 1:
                        # We check if the target qubit and the control qubit both commute through their own wires.
                        # The commutation information (i.e. the found CNOT node which half-commutes with the current
                        # CNOT node) obtained by the search process for both the target and the control wire
                        # must be the same for a successful full-commutation
                        res1 = self._search(wire, control, target, node, type, node_idx)
                        other_wire = self.dag.wires[other_wire]
                        res2 = self._search(other_wire, control, target, node, other_type)
                        if res1 == res2 and res1 is not None:
                            commuted_nodes.append(node)
                            commuted_nodes.append(res1)

            # Remove the CNOT nodes after commutation
            for n in commuted_nodes:
                self.dag.remove_op_node(n)

        return self.dag

    def _search(self, wire_idx, control_idx, target_idx, node, type, index=None):
        """
        Searches for and returns the node which half-commutes with the given node. Returns None if no commutation
        is possible.

        :param wire_idx: The index of the wire which the given CNOT node is on
        :param control_idx: The wire index corresponding to the control qubit of the to-be-commuted node
        :param target_idx: The target index corresponding to the target qubit of the to-be-commuted node
        :param node: The CNOT node to be commuted
        :param type: The type is 'control' if the to-be-commuted node has its control qubit on the inspected wire;
        'target' if its target qubit is on the wire instead
        """

        wire_nodes = list(self.dag.nodes_on_wire(wire_idx, True))
        if index is not None and index + 1 != len(wire_nodes):
            wire_nodes = wire_nodes[index + 1:]
        elif index is None:
            for idx, n in enumerate(wire_nodes):
                if n == node and idx + 1 != len(wire_nodes):
                    wire_nodes = wire_nodes[idx + 1:]
                    break
        else:
            return None

        return self._get_commutation_info(control_idx, target_idx, wire_nodes, type)

    def _get_commutation_info(self, control_idx, target_idx, wire_nodes, type: str):
        """
        Returns the node to be cancelled with if a half-commutation was successful. Returns None otherwise.

        :param control_idx: The wire index corresponding to the control qubit of the to-be-commuted node
        :param target_idx: The target index corresponding to the target qubit of the to-be-commuted node
        :param wire_nodes: The nodes on the wire of the to-be-commuted node
        :param type: The type is 'control' if the to-be-commuted node has its control qubit on the inspected wire;
        'target' if its target qubit is on the wire instead
        """

        cursor = 0
        while cursor < len(wire_nodes):
            current_node = wire_nodes[cursor]
            if current_node.name == "cx":
                control = current_node.qargs[0].index
                target = current_node.qargs[1].index
                type_dic = {"control": control_idx, "target": target_idx}
                current_node_type_dic = {"control": control, "target": target}
                types = ["control", "target"]
                types.remove(type)
                other_type = types[0]
                if current_node_type_dic[type] == type_dic[type] and current_node_type_dic[other_type] != type_dic[
                    other_type]:
                    # Commute
                    cursor += 1
                    continue
                elif current_node_type_dic[type] != type_dic[type]:
                    # Cannot commute
                    return None
                elif target == target_idx and control == control_idx:
                    # There is a possibility for cancellation
                    return current_node
            elif current_node.name == "h" and abs(control_idx - target_idx) == 1:
                # We try to commute with the third commutation rule
                if cursor + 2 != len(wire_nodes):
                    lookahead_nodes = wire_nodes[cursor: cursor + 3]
                    if self._commutes(lookahead_nodes, target_idx):
                        cursor += 3
                        continue
            return None
        return None

    def _commutes(self, lookahead_nodes, target_idx):
        """
        Returns True if the given three nodes correspond to the third commutation rule.

        :param lookahead_nodes: The nodes to be commuted
        :param target_idx: The wire index of the target qubit of the CNOT Gate to be commuted
        """
        names = [node.name for node in lookahead_nodes]
        if names == ["h", "cx", "h"]:
            control = lookahead_nodes[1].qargs[0].index
            target = lookahead_nodes[1].qargs[1].index
            if control == target_idx and target > control:
                return True
        return False
