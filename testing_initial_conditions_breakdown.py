from network_state import network
from mathematical_functions import break_down_init_cons
import sympy as sp

u_values = [0]*network.number_of_neurons
sij_values = []
for i in range(network.number_of_neurons**2):
    sij_values.append(i)
init_con = u_values + sij_values
u, s = break_down_init_cons(init_con)

sp.pprint(u)
print("\n")
sp.pprint(s)

