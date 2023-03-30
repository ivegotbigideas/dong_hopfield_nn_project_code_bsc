from network_state import network
from simulate_network import t, sol
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

for index in range(network.number_of_neurons, 2*network.number_of_neurons):
    plt.plot(t, sol[:,index])

plt.xlabel("$t$", fontsize=17)
plt.ylabel("$s_{%s}$" % "0j", fontsize=17)
plt.grid()
plt.show()
