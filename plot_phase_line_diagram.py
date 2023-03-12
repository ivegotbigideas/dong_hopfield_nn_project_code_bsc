from matplotlib.widgets import Slider
from mathematical_functions import dudt, find_fixed_points
from network_state import network
import matplotlib.pyplot as plt
import numpy as np

# validation
if network.number_of_neurons != 1:
    print("Can only work with 1 neuron! You have: %s neurons." % network.number_of_neurons)
    exit(1)

# setup plot
plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
#fig.subplots_adjust(bottom=0.15)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r'$\frac{du_0}{dt}$', fontsize=18)
ax.xaxis.labelpad = 17
ax.set_ylabel('$u_0$', fontsize=15, rotation=0)
ax.yaxis.labelpad = 17
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("right")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
ax.set_xticks([-6.0, -4.0, -2.0, 2.0, 4.0, 6.0])
ax.set_yticks([-6.0, -4.0, -2.0, 2.0, 4.0, 6.0])

# data preparation functions
def prepare_plotting_data():
    derivatives = []
    for u_instance in u:
        derivatives.append(dudt([u_instance], network.s, network.focal_neurons[0]))
    fixed_points = find_fixed_points()
    return derivatives, fixed_points

# plotting functions
def update_plot(*args):
    #network.g = g_constant_slider.val
    #network.s[network.focal_neurons[0]][network.focal_neurons[0]] = S00_slider.val
    #network.a[network.focal_neurons[0]] = a0_slider.val

    derivatives, fixed_points = prepare_plotting_data()
    line[0].set_ydata(derivatives)
    fp_markers[0].set_xdata(fixed_points)

# prepare data
u = np.linspace(-6,6,5000)
derivatives, fixed_points = prepare_plotting_data()

# create sliders
#g_constant_slider = Slider(plt.axes([0.25, 0.025, 0.65, 0.03]), 'g constant slider', valmin=-10, valmax=10, valinit=network.g, valstep=0.05)
#S00_slider = Slider(plt.axes([0.25, 0.055, 0.65, 0.03]), '$S_{%s}$ slider' % "00", valmin=-15, valmax=15, valinit=network.s[network.focal_neurons[0]][network.focal_neurons[0]], valstep=0.01)
#a0_slider = Slider(plt.axes([0.25, 0.085, 0.65, 0.03]), f'$a_{0}$ slider', valmin=1, valmax=3, valinit=network.a[network.focal_neurons[0]], valstep=0.01)

# add slider behaviour
#g_constant_slider.on_changed(update_plot)
#S00_slider.on_changed(update_plot)
#a0_slider.on_changed(update_plot)

# plot data
line = ax.plot(u, derivatives, zorder=10, linewidth=0.8)
fp_markers = ax.plot(fixed_points, [0,0,0], marker="x", linestyle="", color="r")

# display
plt.show()
