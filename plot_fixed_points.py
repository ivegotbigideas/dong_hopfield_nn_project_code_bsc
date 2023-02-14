from mathematical_functions import find_fixed_points
from simulate_network import sol
from network_state import network
import numpy as np
import matplotlib.pyplot as plt

s = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
fixed_points = find_fixed_points(s)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for fixed_point in fixed_points:
    print(fixed_point)
    fp_plot = plt.scatter(fixed_point[0],fixed_point[1], marker="x")

plt.grid()
plt.show()