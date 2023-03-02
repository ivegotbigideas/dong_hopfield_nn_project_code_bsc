from simulate_network import t, sol
from scipy.integrate import odeint
from mathematical_functions import calculate_network_state
from network_state import network
import numpy as np

u = [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
#u = [1, -1, 1, -1, 1, 1, -1, 1, 1, 1]
#u = [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1]
#u = [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
#u = [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
#u = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1]

sol = sol[999].tolist()

for i in range(network.number_of_neurons):
    sol.pop(i)

u.extend(sol)

path_to_attractor = odeint(calculate_network_state, u, t)
print(path_to_attractor[999][0:network.number_of_neurons-1])