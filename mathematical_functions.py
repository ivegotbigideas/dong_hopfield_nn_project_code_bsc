from math import pi, sqrt
from scipy import optimize
from network_state import network
import numpy as np

def sigmoid(x):
    return 2/pi*np.arctan(1.4*pi*x/2)
