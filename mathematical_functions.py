from math import pi
from scipy import optimize
from network_state import network
import numpy as np

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

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

# TODO: Update guesses when s_id < 0
def find_fixed_points_of_2D_system():
    if network.s[network.focal_neurons[0]][network.focal_neurons[1]] >= 0:
        hor_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[-3],[0],[3]], maxiter=2000)
        vert_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[-3],[0],[3]], maxiter=2000)
    else:
        hor_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[-3],[0],[3]], maxiter=2000)
        vert_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[3],[0],[-3]], maxiter=2000)
    values = []
    for index in range(len(hor_values)):
        values.append([hor_values[index][0], vert_values[index][0]])
    return values

# This function only works in a specific case right now - a 2 neuron system with no external stimulus. It should be generalised
def iterate_2d_sys_fixed_point(u):
    value = network.g * sigmoid(network.s[network.focal_neurons[0]][network.focal_neurons[1]]) * sigmoid(network.g * sigmoid(network.s[network.focal_neurons[1]][network.focal_neurons[0]]) * sigmoid(u))
    return value