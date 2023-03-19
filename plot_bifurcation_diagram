from mathematical_functions import find_fixed_points, test_instability_condition
from network_state import network
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# horizontal axis values
s_values = np.linspace(-1,1,1000)

# setup plot
plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("$s_{00}$", fontsize=17)
ax.set_ylabel("$u_0$ (fixed points)", fontsize=17)
plt.xlim([-1, 1])
plt.ylim([-2, 2])

# prepare data
fp = []
unstable_fp_s_values = []
for s_value in s_values:
    network.s = [[s_value]]
    numerical_fixed_points = find_fixed_points()
    fp.append(numerical_fixed_points)

    if test_instability_condition():
        unstable_fp_s_values.append(s_value)
    else:
        unstable_fp_s_values.append(None)

u0_fixed_points = []
for point in fp:
    u0_fixed_points.append([point[0], point[2]])
# plot data
ax.plot(unstable_fp_s_values, [0]*len(unstable_fp_s_values), '--')
ax.plot(s_values, u0_fixed_points, color="r")

# display
plt.show()