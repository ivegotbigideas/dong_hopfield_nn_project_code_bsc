from mathematical_functions import test_instability_condition
from network_state import network
import numpy as np
import matplotlib.pyplot as plt

# validation
if network.NUMBER_OF_NEURONS != 2:
    print("Can only work with 2 neurons! You have: %s neurons." % network.NUMBER_OF_NEURONS)
    exit(1)
elif network.I != [0, 0]:
    print("Can only work with 0 external stimulus! You have external stimulus: %s." % str(network.I))
    exit(1)

# axis values
g_values = np.linspace(0, 5, 1000)
s_values = np.linspace = np.linspace(0, 20, 10000) # must count from y axis outwards! i.e. must be (0, X, 10000)

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("$g$")
ax.set_ylabel("$s_{%s}$" % (str(network.focal_neurons[0]) + str(network.focal_neurons[1])))

# prepare data
data = []
for g_value in g_values:
    network.g = g_value
    for s_value in s_values:
        network.s = np.array([[0, s_value],[s_value, 0]])
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
ax.plot(x_values, y_values)

# display
plt.show()
