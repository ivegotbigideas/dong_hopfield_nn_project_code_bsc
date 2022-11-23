from matplotlib.widgets import Slider
from mathematical_functions import dudt
from network_state import network
import matplotlib.pyplot as plt
import numpy as np

# validation
if network.number_of_neurons != 1:
    print("Can only work with 1 neuron! You have: %s neurons." % network.number_of_neurons)
    exit(1)

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(f"$du_{network.focal_neurons[0]}/dt$")
ax.set_ylabel(f'$u_{network.focal_neurons[0]}$')
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("right")
plt.xlim([-6, 6])
plt.ylim([-6, 6])

# data preparation functions
def prepare_plotting_data():
    derivatives = []
    for u_instance in u:
        derivatives.append(dudt([u_instance], network.s, network.focal_neurons[0]))
    slope, intercept = np.polyfit(u, derivatives, 1)
    return derivatives, intercept

# plotting functions
def update_plot(*args):
    network.A = A_constant_slider.val
    network.I[network.focal_neurons[0]] = I_slider.val

    derivatives, intercept = prepare_plotting_data()
    line[0].set_ydata(derivatives)
    x_intercept[0].set_xdata(intercept)
    left_arrow[0].set_xdata(intercept - 0.5)
    right_arrow[0].set_xdata(intercept + 0.5)

# prepare data
u = np.linspace(-6,6,5000)
derivatives, intercept = prepare_plotting_data()

# create sliders
A_constant_slider = Slider(plt.axes([0.25, 0.025, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=network.A, valstep=0.05)
I_slider = Slider(plt.axes([0.25, 0.055, 0.65, 0.03]), f'$I_{0}$ slider', valmin=-15, valmax=15, valinit=network.I[network.focal_neurons[0]], valstep=0.01)

# add slider behaviour
A_constant_slider.on_changed(update_plot)
I_slider.on_changed(update_plot)

# plot data
line = ax.plot(u, derivatives)
x_intercept = ax.plot(intercept, 0, "red", marker = "x", markersize = 7.0)
left_arrow = ax.plot(intercept - 0.5, 0, "red", marker = ">", markersize = 7.0)
right_arrow = ax.plot(intercept + 0.5, 0, "red", marker = "<", markersize = 7.0)

# display
plt.show()
