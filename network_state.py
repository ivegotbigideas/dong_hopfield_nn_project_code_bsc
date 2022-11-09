import numpy as np

class Network:
    def __init__(self):
        # basic network information
        self.NUMBER_OF_NEURONS = 2
        self.focal_neurons = [0,1]

        # network state
        self.I = [0, 0]
        self.s = np.array([ [0, 0], [0, 0] ])

        # equation constants
        self.g = 5
        self.a = [1, 1]
        self.A = 2

network = Network()