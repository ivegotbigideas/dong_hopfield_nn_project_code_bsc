from math import e as exp
from matplotlib.widgets import Slider
from mathematical_functions import two_dim_system, find_fixed_points_of_2D_system
from network_state import network
import matplotlib.pyplot as plt
import numpy as np

# data preparation functions
def prepare_derivative_data():
    DU0, DU1 = two_dim_system([U0, U1])
    clrMap = (np.hypot(DU0, DU1))
    clrMap[ clrMap==0 ] = 1
    DU0 /= clrMap
    DU1 /= clrMap
    return DU0, DU1, clrMap

# plotting functions
def update_plot(*args):
    network.g = g_constant_slider.val

    network.A = A_constant_slider.val

    network.a[network.focal_neurons[0]] = a0_constants_slider.val
    network.a[network.focal_neurons[1]] = a1_constants_slider.val

    network.I[network.focal_neurons[0]] = I0_slider.val
    network.I[network.focal_neurons[1]] = I1_slider.val

    network.s[network.focal_neurons[0]][network.focal_neurons[1]] = s_slider.val
    network.s[network.focal_neurons[1]][network.focal_neurons[0]] = s_slider.val

    DU0, DU1, clrMap = prepare_derivative_data()

    # update fixed point data
    fp = find_fixed_points_of_2D_system()
    for marker in range(len(C)):
        C[marker][0].set_data(fp[marker])

    Q.set_UVC(DU0, DU1, clrMap)
    fig.canvas.draw()

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)

# prepare data
u0 = np.linspace(-6,6,20)
u1 = np.linspace(-6,6,20)

U0, U1 = np.meshgrid(u0, u1)
DU0, DU1, clrMap = prepare_derivative_data()

fp = find_fixed_points_of_2D_system()
C = []
for point in fp:
    C.append(ax.plot(point[0], point[1],"red", marker = "x", markersize = 7.0))

# plot quivers
Q = ax.quiver(U0, U1, DU0, DU1, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{network.focal_neurons[0]}$')
ax.set_ylabel(f"$u_{network.focal_neurons[1]}$")
ax.grid()

# create sliders
fig2 = plt.figure()
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=network.g, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=network.A, valstep=0.05)
a0_constants_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'$a_{network.focal_neurons[0]}$ constant slider', valmin=0.1, valmax=1, valinit=network.a[network.focal_neurons[0]], valstep=0.01)
a1_constants_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'$a_{network.focal_neurons[1]}$ constant slider', valmin=0.1, valmax=1, valinit=network.a[network.focal_neurons[1]], valstep=0.01)
I0_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), f'$I_{0}$ slider', valmin=-15, valmax=15, valinit=network.I[network.focal_neurons[0]], valstep=0.01)
I1_slider = Slider(plt.axes([0.25, 0.35, 0.65, 0.03]), f'$I_{1}$ slider', valmin=-15, valmax=15, valinit=network.I[network.focal_neurons[1]], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.4, 0.65, 0.03]), '$s_{%s}$ slider' % (str(network.focal_neurons[0])+str(network.focal_neurons[1])), valmin=-5, valmax=5, valinit=network.s[network.focal_neurons[0]][network.focal_neurons[1]], valstep=0.05)

# add slider behaviour
g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
a0_constants_slider.on_changed(update_plot)
a1_constants_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
I1_slider.on_changed(update_plot)
s_slider.on_changed(update_plot)

# display
plt.show()