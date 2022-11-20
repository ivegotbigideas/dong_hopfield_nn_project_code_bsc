import numpy as np

class Network:
    def __init__(self):
        # basic network information
        self.number_of_neurons = 2
        self.focal_neurons = [0,1] # for plotting so max length should be 2

        # network state
        self.I = [0]*self.number_of_neurons
        self.s = []
        for neuron_id in range(self.number_of_neurons):
            row = [1]*self.number_of_neurons
            row[neuron_id] = 0
            self.s.append(row)
        self.s = np.array(self.s, dtype=np.float64)

        # equation constants
        self.g = 1
        self.a = [1, 1]
        self.A = 1

network = Network()