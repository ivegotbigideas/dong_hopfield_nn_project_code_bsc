from math import pi
from scipy import optimize
import numpy as np

# basic script information
NUMBER_OF_NEURONS = 2
focal_neurons = [0,1]

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

def dudt(u, neuron_id, number_of_neurons, I, s, g, a, A):
    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(number_of_neurons):
        if pointer != neuron_id:
            connection_strength=sigmoid(s[neuron_id, pointer]) # T
            sum += connection_strength * sigmoid(u[pointer])
    term_2 = g*sum

    term_3 = A * I[neuron_id]

    derivative = 1/a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def two_dim_system(u, I, s, g, a, A):
    return np.array([dudt(u, focal_neurons[0], NUMBER_OF_NEURONS, I, s, g, a, A), dudt(u, focal_neurons[1], NUMBER_OF_NEURONS, I, s, g, a, A)])

def find_fixed_points(I, s, g, a, A):
    hor_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[-0.5],[0.01],[0.5]], args=(s, g), maxiter=2000)
    vert_values = optimize.fixed_point(iterate_2d_sys_fixed_point, [[-0.5],[0.01],[0.5]], args=(s, g), maxiter=2000)
    values = []
    for index in range(len(hor_values)):
        values.append([hor_values[index][0], vert_values[index][0]])
    return values

# This function only works in a specific case right now. It should be generalised
def iterate_2d_sys_fixed_point(u, s, g):
    return g*sigmoid(s[focal_neurons[0]][focal_neurons[1]])*sigmoid(u)