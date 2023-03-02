from simulate_network import t, sol
from scipy.integrate import odeint
from mathematical_functions import find_attractors_informally
from network_state import network
import matplotlib.pyplot as plt

#init_con = [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
#init_con = [1, -1, 1, -1, 1, 1, -1, 1, 1, 1]
#init_con = [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1]
init_con = [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
#init_con = [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1]
#init_con = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1]

sol = sol[len(t)-1].tolist()

for i in range(network.number_of_neurons):
    sol.pop(i)

init_con.extend(sol)

path_to_attractor = odeint(find_attractors_informally, init_con, t)
print(path_to_attractor[len(t)-1][0:network.number_of_neurons-1])
for position in path_to_attractor:
    plt.plot(position)

plt.grid()
plt.show()