import numpy as np

class Network:
    def __init__(self):
        # basic network information
        self.number_of_neurons = 1
        self.focal_neurons = [0] # for plotting so max length should be 2

        # network state
        self.I = [0]*self.number_of_neurons
        self.u = [1]*self.number_of_neurons
        self.s = []
        for neuron_id in range(self.number_of_neurons):
            row = [1]*self.number_of_neurons
            row[neuron_id] = 0.1
            self.s.append(row)
        self.s = np.array(self.s, dtype=np.float64)

        # equation constants
        self.g = 5
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