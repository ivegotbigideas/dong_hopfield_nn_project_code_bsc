from scipy.integrate import odeint
from mathematical_functions import find_fixed_points, determine_stability, find_attractors_informally
from network_state import network, refactor_s_vector
import numpy as np
import matplotlib.pyplot as plt

network.s = np.loadtxt("default_s_values.txt")
if False: # set to false to just load default s values
    from simulate_network import sol
    final_s_values = np.ndarray.tolist(sol[len(sol)-1, network.number_of_neurons:len(sol[0])])
    network.s = np.array(refactor_s_vector(final_s_values))
fixed_points = find_fixed_points()

plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(7, 7), dpi=200)
ax = fig.add_subplot(1,1,1)
plt.xlabel("$u_0$", fontsize=17)
plt.ylabel("$u_1$", fontsize=17)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot fixed point
for fixed_point in fixed_points:
    fixed_point = list(fixed_point)

    stability = determine_stability(fixed_point)
    if stability == "stable":
        fp_plot = plt.scatter(fixed_point[0], fixed_point[1], marker="x")
    elif stability == "unstable":
        fp_plot = plt.scatter(fixed_point[0], fixed_point[1], 4, marker="o")
    else:
        fp_plot = plt.scatter(fixed_point[0], fixed_point[1], marker="^")
    
ax.set_aspect('equal', adjustable='box')

# plot trajectories
init_cons = [network.get_I(0),
             network.get_I(12),
             network.get_I(24),
             network.get_I(36),
             network.get_I(48),
             network.get_I(60)
            ]

# for recognisation
init_cons = [init_cons[0]]
for _ in range(1,4):
    disp = np.random.uniform(low=-1, high=1, size=10)
    init_cons.append(list(init_cons[0] + disp))
print(init_cons)

for init_con in init_cons:
    init_con.extend(network.s.flatten())
    t = np.linspace(0, 8*300, 500)
    traj = odeint(find_attractors_informally, init_con, t)

    final_traj = []
    for timestep in traj:
        #print(timestep[0:2])
        final_traj.append(timestep[0:2])

    x = []
    y = []
    for point in final_traj:
        x.append(point[0])
        y.append(point[1])
    if init_con != init_cons[0]:
        plt.plot(x,y,color="black")
    else:
        plt.plot(x,y, zorder=10)

plt.gca().set_aspect('equal')
plt.grid()
plt.savefig('plot.png')
plt.show()