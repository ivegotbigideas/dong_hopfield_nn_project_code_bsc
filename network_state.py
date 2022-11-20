import numpy as np

class Network:
    def __init__(self):
        # basic network information
        self.NUMBER_OF_NEURONS = 2
        self.focal_neurons = [0,1] # for plotting so max length should be 2

        # network state
        self.I = [0]*self.NUMBER_OF_NEURONS
        self.s = []
        for neuron_id in range(self.NUMBER_OF_NEURONS):
            row = [1]*self.NUMBER_OF_NEURONS
            row[neuron_id] = 0
            self.s.append(row)
        self.s = np.array(self.s, dtype=np.float64)

        # equation constants
        self.g = 1
        self.a = [1, 1]
        self.A = 1

network = Network()