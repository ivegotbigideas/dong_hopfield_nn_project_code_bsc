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
            ds_dt_equations.append(dsdt(s, u, neuron_id_1, neuron_id_2))
    return np.array(ds_dt_equations)

def find_fixed_points_of_2D_system(s):
    if (s[network.focal_neurons[0]][network.focal_neurons[1]] >= 0 and network.g>=0) or (s[network.focal_neurons[0]][network.focal_neurons[1]] < 0 and network.g < 0):
        u_inits = [[-3, -3], [0, 0], [3, 3]]
    else:
        u_inits = [[-3, 3], [0, 0], [3, -3]]
    fixed_points = []
    for u_init in u_inits:
        fixed_point = optimize.newton(system_of_dudt_eqns, u_init, maxiter=2000, args=(network.s, ))
        fixed_points.append(fixed_point)
    return fixed_points

def test_instability_condition():
    unstable = False
    s = network.s[network.focal_neurons[0]][network.focal_neurons[1]]
    sig = sigmoid(s)
    if network.g == 0:
        pass
    elif (sig > 5/(7*network.g)) ^ (sig < -5/(7*network.g)):
        unstable = True
    return unstable