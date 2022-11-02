from math import e as exp
from matplotlib.widgets import Slider
from scipy import integrate
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# script constants
NUMBER_OF_NEURONS = 2

# network state
I = [0, 0]
s = np.array([ [0, 1], [1, 0] ])

# equation constants
g = 5
a = [1, 1]
A = 2

# custom constants
Z = 10

# mathematical functions
def sigmoid(x):
    value = 2/pi*np.arctan(1.4*pi*x/2)
    return value

def dudt(neuron_id, number_of_neurons, u, I, s, g, a):
    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(number_of_neurons):
        if pointer != neuron_id:
            connection_strength=sigmoid(s[neuron_id, pointer]) # T
            sum += connection_strength * sigmoid(u[pointer])
    term_2 = g*sum

    term_3 = A * I[neuron_id]

    derivative = 1/a[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def system(u):
    return np.array([dudt(0, NUMBER_OF_NEURONS, u, I, s, g, a), dudt(1, NUMBER_OF_NEURONS, u, I, s, g, a)])

# plotting functions
def update_plot(*args):
    global g
    g = g_constant_slider.val

    global A
    A = A_constant_slider.val

    global a
    a[0] = a0_constants_slider.val
    a[1] = a1_constants_slider.val

    global I
    I[0] = I0_slider.val
    I[1] = I0_slider.val

    global s
    s[0][1] = s_slider.val
    s[1][0] = s_slider.val
    DU1, DU2 = system([U1, U2])
    clrMap = (np.hypot(DU1, DU2))
    clrMap[ clrMap==0 ] = 1
    DU1 /= clrMap
    DU2 /= clrMap

    Q.set_UVC(DU1, DU2)
    fig.canvas.draw()

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)

# plot quivers
u1 = np.linspace(-6,6,20)
u2 = np.linspace(-6,6,20)

U1, U2 = np.meshgrid(u1, u2)
DU1, DU2 = system([U1, U2])
clrMap = (np.hypot(DU1, DU2))
clrMap[ clrMap==0 ] = 1
DU1 /= clrMap
DU2 /= clrMap

Q = ax.quiver(U1, U2, DU1, DU2, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{0}$')
ax.set_ylabel(f"$u_{1}$")
ax.grid()

# create sliders
fig2 = plt.figure()
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A, valstep=0.05)
a0_constants_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'$a_{0}$ constant slider', valmin=0.1, valmax=1, valinit=a[0], valstep=0.01)
a1_constants_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'$a_{1}$ constant slider', valmin=0.1, valmax=1, valinit=a[1], valstep=0.01)
I0_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), f'$I_{0}$ slider', valmin=-15, valmax=15, valinit=I[0], valstep=0.01)
I1_slider = Slider(plt.axes([0.25, 0.35, 0.65, 0.03]), f'$I_{1}$ slider', valmin=-15, valmax=15, valinit=I[1], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.4, 0.65, 0.03]), '$s_{%s}$ slider' % (str(0)+str(1)), valmin=-10, valmax=10, valinit=s[0][1], valstep=0.1)

g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
a0_constants_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
s_slider.on_changed(update_plot)

# display
plt.show()