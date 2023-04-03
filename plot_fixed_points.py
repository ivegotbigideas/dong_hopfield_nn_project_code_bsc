from scipy.integrate import odeint
from mathematical_functions import find_fixed_points, determine_stability, find_attractors_informally
from network_state import network, refactor_s_vector
import numpy as np
import matplotlib.pyplot as plt
"""
from simulate_network import sol
final_s_values = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
network.s = refactor_s_vector(final_s_values)
"""

network.s = np.loadtxt("default_s_values.txt")
fixed_points = find_fixed_points()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot fixed point
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

# plot trajectory
init_con = [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
init_con.extend(network.s.flatten())
t = np.linspace(0, 8*300, 500)
traj = odeint(find_attractors_informally, init_con, t)

final_traj = []
for timestep in traj:
    print(timestep[0:2])
    final_traj.append(timestep[0:2])

x = []
y = []
for point in final_traj:
    x.append(point[0])
    y.append(point[1])
plt.plot(x,y)

plt.grid()
plt.show()