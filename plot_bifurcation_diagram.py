from mathematical_functions import find_fixed_points_of_2D_system, test_instability_condition
from network_state import network
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# horizontal axis values
s_values = np.linspace(-1,1,1000)

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("$s_{%s}$" % (str(network.focal_neurons[0]) + str(network.focal_neurons[1])))
ax.set_ylabel("$u_{%s}$ (fixed points)" % network.focal_neurons[0])
plt.xlim([-1, 1])
plt.ylim([-2, 2])

# prepare data
# TODO: Make more efficient. Add analytically stable fixed point at (0,0)
fp = []
unstable_fp_s_values = []
for s_value in s_values:
    network.s = np.array([[0, s_value],[s_value, 0]])
    numerical_fixed_points = find_fixed_points_of_2D_system()
    fp.append(numerical_fixed_points)

    if test_instability_condition():
        unstable_fp_s_values.append(s_value)
    else:
        unstable_fp_s_values.append(None)

u0_fixed_points = []
for point in fp:
    u0_fixed_points.append([point[0][0], point[2][0]]) #[fixed_point_id][neuron_id]

# plot data
ax.plot(unstable_fp_s_values, [0]*len(unstable_fp_s_values), '--')
ax.plot(s_values, u0_fixed_points, color="r")

# display
plt.show()