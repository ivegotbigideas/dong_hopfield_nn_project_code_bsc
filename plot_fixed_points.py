from mathematical_functions import find_fixed_points
from simulate_network import sol
from network_state import network
import numpy as np

s = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
print(find_fixed_points(s))
