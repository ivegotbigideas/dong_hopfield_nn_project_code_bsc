from math import e as exp
from matplotlib.widgets import Slider
from scipy import integrate
from mathematical_functions import dudt
import matplotlib.pyplot as plt
import numpy as np

# basic script information
NUMBER_OF_NEURONS = 2
focal_neurons = [0,1]

# network state
I = [0, 0]
s = np.array([ [0, 1], [1, 0] ])

# equation constants
g = 5
a = [1, 1]
A = 2

# mathematical functions
def two_dim_system(u):
    return np.array([dudt(focal_neurons[0], NUMBER_OF_NEURONS, u, I, s, g, a, A), dudt(focal_neurons[1], NUMBER_OF_NEURONS, u, I, s, g, a, A)])

# plotting functions
def update_plot(*args):
    global g
    g = g_constant_slider.val

    global A
    A = A_constant_slider.val

    global a
    a[focal_neurons[0]] = a0_constants_slider.val
    a[focal_neurons[1]] = a1_constants_slider.val

    global I
    I[focal_neurons[0]] = I0_slider.val
    I[focal_neurons[1]] = I0_slider.val

    global s
    s[focal_neurons[0]][focal_neurons[1]] = s_slider.val
    s[focal_neurons[1]][focal_neurons[0]] = s_slider.val

    DU0, DU1 = two_dim_system([U0, U1])
    clrMap = (np.hypot(DU0, DU1))
    clrMap[ clrMap==0 ] = 1
    DU0 /= clrMap
    DU1 /= clrMap

    Q.set_UVC(DU0, DU1)
    fig.canvas.draw()

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)

# plot quivers
u0 = np.linspace(-6,6,20)
u1 = np.linspace(-6,6,20)

U0, U1 = np.meshgrid(u0, u1)
DU0, DU1 = two_dim_system([U0, U1])
clrMap = (np.hypot(DU0, DU1))
clrMap[ clrMap==0 ] = 1
DU0 /= clrMap
DU1 /= clrMap

Q = ax.quiver(U0, U1, DU0, DU1, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{focal_neurons[0]}$')
ax.set_ylabel(f"$u_{focal_neurons[1]}$")
ax.grid()

# create sliders
fig2 = plt.figure()
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A, valstep=0.05)
a0_constants_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'$a_{focal_neurons[0]}$ constant slider', valmin=0.1, valmax=1, valinit=a[focal_neurons[0]], valstep=0.01)
a1_constants_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'$a_{focal_neurons[1]}$ constant slider', valmin=0.1, valmax=1, valinit=a[focal_neurons[1]], valstep=0.01)
I0_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), f'$I_{0}$ slider', valmin=-15, valmax=15, valinit=I[focal_neurons[0]], valstep=0.01)
I1_slider = Slider(plt.axes([0.25, 0.35, 0.65, 0.03]), f'$I_{1}$ slider', valmin=-15, valmax=15, valinit=I[focal_neurons[1]], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.4, 0.65, 0.03]), '$s_{%s}$ slider' % (str(focal_neurons[0])+str(focal_neurons[1])), valmin=-10, valmax=10, valinit=s[focal_neurons[0]][focal_neurons[1]], valstep=0.1)

g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
a0_constants_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
s_slider.on_changed(update_plot)

# display
plt.show()