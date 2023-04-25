from mathematical_functions import test_instability_condition
from network_state import network
import numpy as np
import matplotlib.pyplot as plt

# setup plot
plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("$g$", fontsize=17)
ax.set_ylabel("$s_{%s}$" % (str(network.focal_neurons[0]) + str(network.focal_neurons[0])), fontsize=17)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-17, 17)

def main(g_values, s_values):
    # prepare data
    data = []
    for g_value in g_values:
        network.g = g_value
        for s_value in s_values:
            network.s = np.array([[s_value]])
            bifurcation = test_instability_condition()
            if bifurcation == True:
                data.append([g_value, s_value])
                break

    x_values = []
    y_values = []
    for point in data:
        x_values.append(point[0])
        y_values.append(point[1])

    # plot data
    ax.plot(x_values, y_values, color="b")

# axis values
g_values_arr = [np.linspace(0, 5, 1000), np.linspace(0, -5, 1000)]
s_values_arr = [np.linspace(0, 20, 10000), np.linspace(-20, 0, 10000)] # Must count from y axis outwards! i.e. must be (0, X, 10000)
for index in [0,1]:
        main(g_values_arr[index], s_values_arr[index])

# display
plt.show()
