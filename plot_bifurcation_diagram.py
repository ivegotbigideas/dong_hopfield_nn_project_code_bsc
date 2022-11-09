import numpy as np
import matplotlib.pyplot as plt
from mathematical_functions import find_fixed_points_of_2D_system
from network_state import I, g, a, A, focal_neurons

# horizontal axis values
s_values = np.linspace(0,4,1000)

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("$s_{%s}$" % (str(focal_neurons[0]) + str(focal_neurons[1])))
ax.set_ylabel("$u_{%s}$ (fixed points)" % focal_neurons[0])

# prepare data
fp = []
for s_value in s_values:
    s_matrix = np.array([[0, s_value],[s_value, 0]])
    fp.append(find_fixed_points_of_2D_system(I, s_matrix, g, a, A))

u0_fps = []
for point in fp:
    u0_fps.append([point[0][0], point[1][0], point[2][0]])

for element in range(len(u0_fps)):
    print(str(element) + ": " + str(u0_fps[element]))

# plot data
ax.plot(s_values, u0_fps, color="r")

# display
plt.show()