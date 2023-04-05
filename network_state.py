import numpy as np
from random import randint
from math import floor

class Network:
    def __init__(self):
        # basic network information
        self.number_of_neurons = 10

        # graphing info
        self.focal_neurons = [0,1] # for plotting so max length should be 2

        # network state
        self.s = self._generate_s_matrix()

        # external stimulus
        self._num_stim_vectors = 6
        self._possible_stimulus_states = self._generate_possible_stimulus_states()
        #self.I = self._possible_stimulus_states[0]

        # equation constants
        self.g = 5
        self.a = [1]*self.number_of_neurons
        self.A = 3
        self.H = 1
        self.B = self._generate_B_matrix()

    def _generate_B_matrix(self):
        B = []
        for neuron_id in range(self.number_of_neurons):
            row = [700]*self.number_of_neurons
            row[neuron_id] = 0
            B.append(row)
        return B

    def _generate_possible_stimulus_states(self):
        possible_stimulus_states = []
        for _ in range(self._num_stim_vectors):
            possible_state = []
            for _ in range(self.number_of_neurons):
                neuron_stimulus = randint(0,1)
                if neuron_stimulus == 0: neuron_stimulus = -1
                possible_state.append(neuron_stimulus)
            possible_stimulus_states.append(possible_state)
        return possible_stimulus_states

    def _generate_s_matrix(self):
        s = []
        for neuron_id in range(self.number_of_neurons):
            row = [0]*self.number_of_neurons
            row[neuron_id] = 0
            s.append(row)
        s = np.array(s, dtype=np.float64)
        return s

    # TODO: Make this function not gross
    def get_I(self, t):
        if t != None:
            if t<0:
                print("t<0!")
                exit(1)
            elif 0 <= t < 12:
                return self._possible_stimulus_states[0]
                #return [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
            elif 12 <= t < 24:
                return self._possible_stimulus_states[1]
                #return [1, -1, 1, -1, 1, 1, -1, 1, 1, 1]
            elif 24 <= t < 36:
                return self._possible_stimulus_states[2]
                #return [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1]
            elif 36 <= t < 48:
                return self._possible_stimulus_states[3]
                #return [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
            elif 48 <= t < 60:
                return self._possible_stimulus_states[4]
                #return [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
            elif 60 <= t < 72:
                return self._possible_stimulus_states[5]
                #return [1, 1, 1, 1, -1, 1, -1, 1, -1, -1]
            elif t == 72:
                return self.get_I(0)
            elif t > 72:
                return self.get_I(t-72)
        else:
            return [0]*self.number_of_neurons

def refactor_state_vector(init_cons):
    split = np.array_split(init_cons, network.number_of_neurons+1)
    u = split[0]
    s = split[1:network.number_of_neurons+1]
    return u, s

def refactor_s_vector(s):
    split = np.array_split(s, network.number_of_neurons)
    return split

network = Network()
