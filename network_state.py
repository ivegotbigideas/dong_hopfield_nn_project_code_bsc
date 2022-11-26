import numpy as np
from random import randint

class Network:
    def __init__(self):
        # basic network information
        self.number_of_neurons = 10
        self.focal_neurons = [0,1] # for plotting so max length should be 2

        # network state
        self.u = [0]*self.number_of_neurons
        self.s = []
        for neuron_id in range(self.number_of_neurons):
            row = [1]*self.number_of_neurons
            row[neuron_id] = 0
            self.s.append(row)
        self.s = np.array(self.s, dtype=np.float64)

        # external stimulus
        self.possible_stimulus_states = []
        for _ in range(6): # the 6 defines that there are 6 possible states
            possible_state = []
            for neuron_id in range(self.number_of_neurons):
                neuron_stimulus = randint(0,1)
                if neuron_stimulus == 0: neuron_stimulus = -1
                possible_state.append(neuron_stimulus)
            self.possible_stimulus_states.append(possible_state)
        self.I = self.possible_stimulus_states[0]

        # equation constants
        self.g = 1
        self.a = [1]*self.number_of_neurons
        self.A = 1
        self.H = 1
        self.B = []
        for neuron_id in range(self.number_of_neurons):
            row = [1]*self.number_of_neurons
            row[neuron_id] = 0
            self.B.append(row)
        self.B = np.array(self.B, dtype=np.float64)

network = Network()
