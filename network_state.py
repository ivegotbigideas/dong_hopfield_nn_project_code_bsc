import numpy as np
from random import randint

class Network:
    def __init__(self):
        # basic network information
        self.number_of_neurons = 10
        self.focal_neurons = [0,1] # for plotting so max length should be 2

        # network state
        self.u = [0]*self.number_of_neurons
        self.s = self.generate_s_matrix(self.number_of_neurons)

        # external stimulus
        _possible_stimulus_states = self.generate_possible_stimulus_states(self.number_of_neurons)
        self.I = _possible_stimulus_states[0]

        # equation constants
        self.g = 1
        self.a = [1]*self.number_of_neurons
        self.A = 1
        self.H = 1
        self.B = self.generate_B_matrix(self.number_of_neurons)

    def generate_B_matrix(self, number_of_neurons):
        B = []
        for neuron_id in range(number_of_neurons):
            row = [1]*number_of_neurons
            row[neuron_id] = 0
            B.append(row)
        return B

    def generate_possible_stimulus_states(self, number_of_neurons):
        possible_stimulus_states = []
        for _ in range(6): # the 6 defines that there are 6 possible states
            possible_state = []
            for _ in range(number_of_neurons):
                neuron_stimulus = randint(0,1)
                if neuron_stimulus == 0: neuron_stimulus = -1
                possible_state.append(neuron_stimulus)
            possible_stimulus_states.append(possible_state)
        return possible_stimulus_states

    def generate_s_matrix(self, number_of_neurons):
        s = []
        for neuron_id in range(number_of_neurons):
            row = [1]*number_of_neurons
            row[neuron_id] = 0
            s.append(row)
        s = np.array(s, dtype=np.float64)
        return s

network = Network()
