import numpy as np
import networkx as nx
from qiskit_optimization.applications import Tsp
from qiskit_optimization.exceptions import QiskitOptimizationError
import matplotlib.pyplot as plt
import dimod
from dimod import ConstrainedQuadraticModel
from tabulate import tabulate

########################################################################################################################
# TSP UTILS
########################################################################################################################

def parse_tsplib_format(filename: str) -> "Tsp":

    coord = []  # type: ignore
    with open(filename, encoding="utf8") as infile:
        coord_section = False
        for line in infile:
            if line.startswith("NAME"):
                name = line.split(":")[1]
                name.strip()
            elif line.startswith("TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "TSP" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "TSP" type. Actual: {typ}'
                    )
            elif line.startswith("DIMENSION"):
                dim = int(line.split(":")[1])
                coord = np.zeros((dim, 2))  # type: ignore
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "EUC_2D" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "EUC_2D" edge weight. Actual: {typ}'
                    )
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
            elif coord_section:
                v = line.split()
                index = int(v[0]) - 1
                coord[index][0] = float(v[1])
                coord[index][1] = float(v[2])

    x_max = max(coord_[0] for coord_ in coord)
    x_min = min(coord_[0] for coord_ in coord)
    y_max = max(coord_[1] for coord_ in coord)
    y_min = min(coord_[1] for coord_ in coord)

    pos = {i: (coord_[0], coord_[1]) for i, coord_ in enumerate(coord)}

    graph = nx.random_geometric_graph(
        len(coord), np.hypot(x_max - x_min, y_max - y_min) + 1, pos=pos
    )

    for w, v in graph.edges:
        delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
        graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))
    return Tsp(graph)

def tsp_value(z: list[int], adj_matrix: np.ndarray) -> float:

    ret = 0.0
    for i in range(len(z) - 1):
        ret += adj_matrix[z[i], z[i + 1]]
    ret += adj_matrix[z[-1], z[0]]
    return ret

def decode_solution_cqm(assignment_dict):
    num_positions = max(int(key.split('_')[2]) for key in assignment_dict.keys())
    output_array = np.zeros((num_positions,), dtype=int)

    for key, value in assignment_dict.items():
        if value == np.float64(1.0):
            _, i, j = key.split('_')
            city = int(i)
            position = int(j)
            output_array[position - 1] = city  # Subtract 1 from position to make it 0-indexed
    output_array = np.insert(output_array, 0, 0)
    return output_array

def print_cqm_stats(cqm: ConstrainedQuadraticModel) -> None:

    # Check if cqm is a dimod.ConstrainedQuadraticModel
    if not isinstance(cqm, ConstrainedQuadraticModel):
        raise ValueError("input instance should be a dimod CQM model")

    # Count the amount of binary, integer and continuous variables in cqm
    num_binaries = sum(cqm.vartype(v) is dimod.BINARY for v in cqm.variables)
    num_continuous = sum(cqm.vartype(v) is dimod.REAL for v in cqm.variables)

    # Count the amount of discrete (one-hot), linear and quadratic constraints
    num_discretes = len(cqm.discrete)
    num_linear_constraints = sum(
        constraint.lhs.is_linear() for constraint in cqm.constraints.values())
    num_quadratic_constraints = sum(
        not constraint.lhs.is_linear() for constraint in
        cqm.constraints.values())

    # Count the amount of "less (greater) or equal than" inequalities and
    # equalities constraints
    num_le_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Le for constraint in
        cqm.constraints.values())
    num_ge_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Ge for constraint in
        cqm.constraints.values())
    num_equality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Eq for constraint in
        cqm.constraints.values())

    # Check that the sum of binaries and continuous variables is equal
    # to the total amount of variables
    assert num_binaries + num_continuous == len(cqm.variables)
    # Check that the sum of linear and quadratic constraints is equal to the
    # total amount of constraints
    assert (num_quadratic_constraints + num_linear_constraints ==
            len(cqm.constraints))

    # Plain text table containing all the information aforementioned
    print(" \n" + "=" * 35 + "MODEL INFORMATION" + "=" * 35)
    print(
        ' ' * 10 + 'Variables' + " " * 20 + 'Constraints' + " " * 15 +
        'Sensitivity')
    print('-' * 30 + " " + '-' * 28 + ' ' + '-' * 18)
    print(tabulate([["Binary", "Continuous", "Quad", "Linear",
                     "One-hot", "EQ  ", "LT", "GT"],
                    [num_binaries, num_continuous,
                     num_quadratic_constraints,
                     num_linear_constraints, num_discretes,
                     num_equality_constraints,
                     num_le_inequality_constraints,
                     num_ge_inequality_constraints]],
                   headers="firstrow"))

def filter_unique_integer_sublists(input_list):
    filtered = []
    for sublist in input_list:
        if all(isinstance(item, int) for item in sublist) and len(sublist) == len(set(sublist)):
            filtered.append(sublist)
    return filtered

########################################################################################################################
# KP UTILS
########################################################################################################################

def parse_kp_format(file):
    data = open(file)
    count = 0
    values = []
    weights = []
    for line in data:
        if count == 2:
            max_weight = int(line)
        if count > 3:
            values.append(int(line.split(" ")[0].strip()))
            weights.append(int(line.split(" ")[1].strip()))
        count = count + 1
    data.close()
    return values, weights, max_weight


def get_max_value_under_weight(values, weights, max_weight):
    # Combine values, weights, and their original indices into a list of tuples
    value_weight_index_pairs = [(value, weight, idx) for idx, (value, weight) in enumerate(zip(values, weights))]

    # Sort the list of tuples by value in descending order
    value_weight_index_pairs.sort(reverse=True, key=lambda x: x[0])

    # Iterate through the sorted value-weight-index pairs
    for value, weight, index in value_weight_index_pairs:
        if weight <= max_weight:
            return value, weight, index  # Return value, weight, and the index in the original list

    return None  # Return None if no value satisfies the weight condition

########################################################################################################################
# MC UTILS
########################################################################################################################

def create_adjacency_matrix(graph, num_nodes=None):
    if num_nodes is None:
        # Infer number of nodes from the highest node index in edges
        num_nodes = max(max(u, v) for u, v in graph.edges) + 1

    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Fill the adjacency matrix
    for u, v in graph.edges:
        adj_matrix[u, v] = graph[u][v]['weight']

    return adj_matrix


def create_graph_from_file(file_path: str):
    g = nx.DiGraph()

    # Open the text file

    with open(file_path, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace and split by spaces
            parts = line.strip().split()

            # Extract nodes and weight from the line
            if len(parts) == 3:
                node1 = int(parts[0]) - 1
                node2 = int(parts[1]) - 1
                weight = float(parts[2])  # Assuming weight is a float, adjust if it's different

                # Add the edge to the graph
                g.add_edge(node1, node2, weight=weight)
    return g


def convert_to_bit_list(bit_dict):
    # Sort the keys based on the numeric value after 'x_'
    sorted_keys = sorted(bit_dict, key=lambda x: int(x.split('_')[1]))

    # Extract the values based on the sorted keys
    bit_list = [int(bit_dict[key]) for key in sorted_keys]

    return bit_list

pass

########################################################################################################################
# Method: information
# Description: this method prints information about the ising and the formulation of the problem
########################################################################################################################

def information(qp, qubo):
    print(qp.prettyprint())
    qubitOp, offset = qubo.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))

########################################################################################################################
# PLOTTING FUNCTIONS
########################################################################################################################

def draw_tsp_solution(G, order):
    colors = ["r" for node in G.nodes]
    pos = [G.nodes[node]["pos"] for node in G.nodes]
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)
    plt.show()

def draw_mc_graph(G):
    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()