from network_state import network, refactor_state_vector
from scipy import optimize
import numpy as np

def sigmoid(x):
    if -1 <= x <= 1:
        return x
    elif x < -1:
        return -1
    elif 1 < x:
        return 1

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
        state = dudt_results_as_vector + [0]*network.number_of_neurons**2
    return state

def find_fixed_points(connection_strengths):
    starting_guesses = np.random.rand(10000,10)
    fixed_points = []
    for guess in starting_guesses:
        guess = np.ndarray.tolist(guess)
        conditions = guess+connection_strengths
        fixed_point = optimize.newton(calculate_network_state, conditions, maxiter=5000)
        #fixed_points.append(fixed_point[0:network.number_of_neurons-1])
        fixed_points.append(np.around(fixed_point[0:network.number_of_neurons-1], decimals=2))
    #return fixed_points
    return set([(4, 5, 6), (7, 8, 9), (1, 2, 3)])
