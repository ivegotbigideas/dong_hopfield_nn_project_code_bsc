import numpy as np
from mathematical_functions import sigmoid

# script settings
MAX_ITER = 1000

# network state
s = np.array([ [0, 1], [1, 0] ])
u1 = 3

# constants
g = 5

def return_u0(g, s, u1):
    u0 = -g * sigmoid(s[0][1]) * sigmoid(u1)
    return u0


