from math import pi, sqrt
from scipy import optimize
from network_state import network, break_down_init_cons
import numpy as np

# def sigmoid(x):
#     return 2/pi*np.arctan(1.4*pi*x/2)

def sigmoid(x):
    if -1 <= x <= 1:
        return x
    elif x < -1:
        return -1
    elif 1 < x:
        return 1

# u, s, neuron_id
def dudt(conditions, t, neuron_id):
    u, s = break_down_init_cons(conditions)

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
    u, s = break_down_init_cons(conditions)
    
    term_1 = -s[neuron_id_1][neuron_id_2]
    term_2 = network.H*sigmoid(u[neuron_id_1])*sigmoid(u[neuron_id_2])
    derivative = (1/network.B[neuron_id_1][neuron_id_2])*(term_1 + term_2)
    return derivative

# u, t, s
def system_of_dudt_eqns(conditions, t):
    du_dt_equations = []
    all_neuron_ids = range(network.number_of_neurons)
    for neuron_id_1 in all_neuron_ids:
        du_dt_equations.append(dudt(conditions, t, neuron_id_1))
    return np.array(du_dt_equations)

# s, u
def system_of_dsdt_eqns(conditions):    
    ds_dt_equations = []
    all_neuron_ids = range(network.number_of_neurons)
    for neuron_id_1 in all_neuron_ids:
        ds_dt_row = []
        for neuron_id_2 in all_neuron_ids:
            if neuron_id_1 != neuron_id_2:
                ds_dt_row.append(dsdt(conditions, neuron_id_1, neuron_id_2))
            else:
                ds_dt_row.append(0.0)
        ds_dt_equations.append(ds_dt_row)
    return np.array(ds_dt_equations)

# u, t, s
def simulate_network(conditions, t):
    dudt_eqns = system_of_dudt_eqns(conditions, t)
    dsdt_eqns = system_of_dsdt_eqns(conditions)
 
    dudt_eqns = np.ndarray.tolist(dudt_eqns)

    dsdt_as_vector = []
    for row in range(len(dsdt_eqns)):
        for element in dsdt_eqns[row]:
            dsdt_as_vector.append(element)
    state = dudt_eqns + dsdt_as_vector
    return state
