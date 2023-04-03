from network_state import network, refactor_state_vector
from scipy import optimize
from math import pi
from numpy.linalg import eig
import numpy as np

"""
MODELLING OF THE SYSTEM
"""

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

# u, s, neuron_id
def dudt(conditions, t, neuron_id):
    u, s = refactor_state_vector(conditions)

    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(network.number_of_neurons):
            if pointer != neuron_id:
                connection_strength=sigmoid(s[neuron_id][pointer]) # T
                sum += connection_strength * sigmoid(u[pointer])
    term_2 = network.g*sum

    term_3 = network.A * network.get_I(t)[neuron_id]

    derivative = 1/network.a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

# s, u, neuron_id_1, neuron_id_2
def dsdt(conditions, neuron_id_1, neuron_id_2):
    u, s = refactor_state_vector(conditions)
    term_1 = -s[neuron_id_1][neuron_id_2]
    term_2 = network.H*sigmoid(u[neuron_id_1])*sigmoid(u[neuron_id_2])
    derivative = 1/network.B[neuron_id_1][neuron_id_2] * (term_1 + term_2)
    return derivative

# u, t, s
def system_of_dudt_eqns(conditions, t):
    dudt_results = []
    all_neuron_ids = range(network.number_of_neurons)
    for neuron_id_1 in all_neuron_ids:
        dudt_results.append(dudt(conditions, t, neuron_id_1))
    return np.array(dudt_results)

# s, u
def system_of_dsdt_eqns(conditions):    
    dsdt_results = []
    for neuron_id_1 in range(network.number_of_neurons):
        dsdt_row = []
        for neuron_id_2 in range(network.number_of_neurons):
            if neuron_id_1 != neuron_id_2:
                dsdt_row.append(dsdt(conditions, neuron_id_1, neuron_id_2))
            else:
                dsdt_row.append(0.0)
        dsdt_results.append(dsdt_row)
    return np.array(dsdt_results)

# u, t, s
def calculate_network_state(conditions, t=None):
    dudt_results = system_of_dudt_eqns(conditions, t)
    
    # put dudt results in list
    dudt_results_as_vector = np.ndarray.tolist(dudt_results)

    if t != None:
        dsdt_results = system_of_dsdt_eqns(conditions)

        # put dsdt results in list
        dsdt_results_as_vector = []
        for row in range(len(dsdt_results)):
            for element in dsdt_results[row]:
                dsdt_results_as_vector.append(element)
        state = dudt_results_as_vector + dsdt_results_as_vector
    else:
        state = dudt_results_as_vector + [0]*network.number_of_neurons**2 # the second term indicates no change in connection weights
    return state

"""
FINDING OF ATTRACTORS
"""

def find_attractors_informally(conditions, t):
    t = None # ensures no external stimulus is applied
    dudt_results = system_of_dudt_eqns(conditions, t)
    
    # put dudt results in list
    dudt_results_as_vector = np.ndarray.tolist(dudt_results)

    state = dudt_results_as_vector + [0]*network.number_of_neurons**2 # the second term indicates no change in connection weights
    return state

def find_fixed_points():
    starting_guesses = np.random.uniform(low=-5,high=5, size=(500,network.number_of_neurons))
    starting_guesses = starting_guesses.tolist()
    starting_guesses.append([0]*network.number_of_neurons)
    fixed_points = []
    for guess in starting_guesses:
        fixed_point = optimize.root(find_fixed_point_proxy, guess, tol=1.48e-08, method='hybr', jac=get_linearisation_matrix)

        add_new_fixed_point = True
        for existing_fp in fixed_points:
            if (norm_of_evaluated_point(existing_fp - fixed_point.x) < 1) and (list(fixed_point.x) != [0]*network.number_of_neurons):
                add_new_fixed_point = False

        if norm_of_evaluated_point(fixed_point.x) > 1e-8:
            add_new_fixed_point = False

        if add_new_fixed_point:
            fixed_points.append(fixed_point.x)

    print("Number of fixed points: " + str(len(fixed_points)) + "\n")
    fixed_points = set(tuple(row) for row in fixed_points)
    return fixed_points

def get_linearisation_matrix(fixed_point):
    linearisation_matrix = np.zeros(shape=(network.number_of_neurons, network.number_of_neurons))
    for row_id in range(network.number_of_neurons):
        for col_id in range(network.number_of_neurons):
            linearisation_matrix[row_id][col_id] = partial_deriv_dudt(fixed_point, row_id, col_id)
    return linearisation_matrix

def determine_stability(fixed_point):
    print("Fixed point: " + str(fixed_point))
    evaluation = evaluate_fixed_point(np.array(fixed_point))
    print("Evaluation: " + str(evaluation))
    print("Distance of evaluation from origin: " + str(np.linalg.norm(evaluation)))

    linearisation_matrix = get_linearisation_matrix(fixed_point)
    
    eigenvalues,eigenvectors = eig(linearisation_matrix)
    print("Eigenvalues: " + str(eigenvalues))

    stability = "unknown"
    if all(np.real(eigenvalue) < 0 for eigenvalue in eigenvalues):
        stability = "stable"
    elif any(np.real(eigenvalue) > 0 for eigenvalue in eigenvalues):
        stability = "unstable"
    print("Stability: " + stability)
    print("\n")
    return stability

def find_fixed_point_proxy(guess):
    guess = guess.tolist()
    for row in network.s:
        for val in row:
            guess.append(val)
    
    return calculate_network_state(guess)[0:network.number_of_neurons]

def evaluate_fixed_point(fp):
    evaluation = find_fixed_point_proxy(fp)
    return evaluation

def norm_of_evaluated_point(point):
    return np.linalg.norm(evaluate_fixed_point(point))

def derivative_of_sigmoid(x):
    return 140/(100+49*(pi*x)**2)
    
def partial_deriv_dudt(u, neuron_id, wrt_id):
    if neuron_id == wrt_id:
        partial_deriv = -1/network.a[neuron_id]
    else:
        partial_deriv = network.g/network.a[neuron_id] * (sigmoid(network.s[neuron_id][wrt_id])*derivative_of_sigmoid(u[wrt_id]))
    return partial_deriv
