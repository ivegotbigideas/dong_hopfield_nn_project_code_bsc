from scipy.integrate import odeint
from network_state import network
from mathematical_functions import calculate_network_state
import numpy as np

u_values = [0]*network.number_of_neurons
sij_values = [0]*(network.number_of_neurons**2)
init_con = u_values + sij_values
t = np.linspace(0, 8*network.B[0][1], 1000)
sol = odeint(calculate_network_state, init_con, t)

print(sol.shape)