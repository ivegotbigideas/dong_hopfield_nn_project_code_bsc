from math import pi
import numpy as np

# basic script information
NUMBER_OF_NEURONS = 2
focal_neurons = [0,1]

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)

def dudt(neuron_id, number_of_neurons, u, I, s, g, a, A):
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
    return np.array([dudt(focal_neurons[0], NUMBER_OF_NEURONS, u, I, s, g, a, A), dudt(focal_neurons[1], NUMBER_OF_NEURONS, u, I, s, g, a, A)])