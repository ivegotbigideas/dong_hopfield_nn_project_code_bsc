from scipy.integrate import odeint, solve_ivp
from network_state import network
from mathematical_functions import simulate_network
import numpy as np
import matplotlib.pyplot as plt

u_values = [0]*network.number_of_neurons
sij_values = [0]*(network.number_of_neurons**2)
init_con = u_values + sij_values
t = np.linspace(0, 8*network.B[0][1], 1000)
sol = odeint(simulate_network, init_con, t)

for index in range(network.number_of_neurons, 2*network.number_of_neurons):
    plt.plot(t, sol[:,index])

plt.xlabel("t")
plt.ylabel("$s_{%s}$" % "0j")
plt.grid()
plt.show()