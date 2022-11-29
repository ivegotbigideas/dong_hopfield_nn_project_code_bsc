from math import pi, sqrt
from scipy import optimize
from network_state import network
import numpy as np

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

def dudt(u, s, neuron_id):
    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(network.number_of_neurons):
        connection_strength=sigmoid(s[neuron_id, pointer]) # T
        sum += connection_strength * sigmoid(u[pointer])
    term_2 = network.g*sum

    term_3 = network.A * network.I[neuron_id]

    derivative = 1/network.a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def dsdt(s, u, neuron_id_1, neuron_id_2):
    term_1 = -s[neuron_id_1][neuron_id_2]
    term_2 = network.H*sigmoid(u[neuron_id_1])*sigmoid(u[neuron_id_2])
    derivative = (1/network.B[neuron_id_1][neuron_id_2])*(term_1 + term_2)
    return derivative

def system_of_dudt_eqns(u, s):
    du_dt_equations = []
    all_neuron_ids = range(network.number_of_neurons)
    for neuron_id_1 in all_neuron_ids:
        du_dt_equations.append(dudt(u, s, neuron_id_1))
    return np.array(du_dt_equations)

def system_of_dsdt_eqns(s, u):
    ds_dt_equations = []
    all_neuron_ids = range(network.number_of_neurons)
    for neuron_id_1 in all_neuron_ids:
        for neuron_id_2 in all_neuron_ids:
            if neuron_id_1 != neuron_id_2:
                ds_dt_equations.append(dsdt(s, u, neuron_id_1, neuron_id_2))
    return np.array(ds_dt_equations)
