from mathematical_functions import find_fixed_points, determine_stability
from simulate_network import sol
from network_state import network
import numpy as np
import matplotlib.pyplot as plt

s_values = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
for i in range(network.number_of_neurons):
    for j in range(network.number_of_neurons):
        network.s[i][j] = s_values[network.number_of_neurons*i+j]

fixed_points = find_fixed_points()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for fixed_point in fixed_points:
    fixed_point = list(fixed_point)

    stability = determine_stability(fixed_point)
    if stability == "stable":
        fp_plot = plt.scatter(fixed_point[0],fixed_point[1], marker="x")
    elif stability == "unstable":
        fp_plot = plt.scatter(fixed_point[0],fixed_point[1], marker="o")
    else:
        fp_plot = plt.scatter(fixed_point[0],fixed_point[1], marker="^")
    
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()