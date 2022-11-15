from math import pi, sqrt
from scipy import optimize
from network_state import network
import numpy as np

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

def inverse_sigmoid(x):
    return 2/(1.4*pi)*np.tan(pi*x/2)

def dudt(u, neuron_id):
    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(network.NUMBER_OF_NEURONS):
        if pointer != neuron_id:
            connection_strength=sigmoid(network.s[neuron_id, pointer]) # T
            sum += connection_strength * sigmoid(u[pointer])
    term_2 = network.g*sum

    term_3 = network.A * network.I[neuron_id]

    derivative = 1/network.a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def two_dim_system(u):
    return np.array([dudt(u, network.focal_neurons[0]), dudt(u, network.focal_neurons[1])])

def find_fixed_points_of_2D_system():
    if network.s[network.focal_neurons[0]][network.focal_neurons[1]] >= 0:
        u_inits = [[-3, -3], [3, 3]]
    else:
        u_inits = [[-3, 3], [3, -3]]
    fixed_points = []
    for u_init in u_inits:
        fixed_point = optimize.newton(two_dim_system, u_init, maxiter=2000)
        fixed_points.append(fixed_point)
    return fixed_points

def test_instability_condition(s_value):
    unstable = False
    if network.g == 0:
        unstable = True
    elif s_value > inverse_sigmoid(5/(7*network.g*sqrt(network.a[network.focal_neurons[0]]*network.a[network.focal_neurons[1]]))):
        unstable = True
    elif s_value < inverse_sigmoid(-5/(7*network.g*sqrt(network.a[network.focal_neurons[0]]*network.a[network.focal_neurons[1]]))):
        unstable = True
    return unstable
