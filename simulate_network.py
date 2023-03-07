from scipy.integrate import odeint
from network_state import network, refactor_s_vector
from mathematical_functions import calculate_network_state
import numpy as np

u_values = [0]*network.number_of_neurons
sij_values = [0]*(network.number_of_neurons**2)
init_con = u_values + sij_values
t = np.linspace(0, 8*300, 500)
sol = odeint(calculate_network_state, init_con, t)

save_new_s_matrix = False
if save_new_s_matrix:
    final_s_values = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
    np.savetxt("default_s_values.txt", refactor_s_vector(final_s_values))
