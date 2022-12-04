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

def dudt_fp(u, s, neuron_id):
    term_1 = -u

    sum = 0
    for pointer in range(network.number_of_neurons):
        connection_strength=sigmoid(s[neuron_id, pointer]) # T
        sum += connection_strength * sigmoid(u)
    term_2 = network.g*sum

    term_3 = network.A * network.I[neuron_id]

    derivative = 1/network.a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def find_fixed_points():
    u_inits = [-3, 0, 3]
    fixed_points = []
    for u_init in u_inits:
        fixed_point = optimize.newton(dudt_fp, u_init, args=(network.s, 0), maxiter=2000)
        fixed_points.append(fixed_point)
    return fixed_points
