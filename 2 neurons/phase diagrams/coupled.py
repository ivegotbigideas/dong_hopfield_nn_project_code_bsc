from math import e as exp
from matplotlib.widgets import Slider
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

# script constants
NUMBER_OF_NEURONS = 2
i = 0
j = 1

# network state
I = [0, 0]
s = np.array([ [0, 0.5], [0.5, 0] ])

# equation constants
g = 5
a = [1, 1]
A = 2

# custom constants
Z = 10

# mathematical functions
def sigmoid(x):
    value = 1/(1+exp**(-x))
    value = value - 0.5
    value = value*2
    return value

def dudt(neuron_id, number_of_neurons, u, I, s, g, a):
    term_1 = -u[neuron_id]

    sum = 0
    for pointer in range(number_of_neurons-1):
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
    a[i] = a0_constants_slider.val
    a[j] = a1_constants_slider.val

    global I
    I[i] = I0_slider.val
    I[j] = I0_slider.val

    global s
    s[i][j] = s_slider.val
    s[j][i] = s_slider.val

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
fig.subplots_adjust(bottom=0.5, hspace=0.5)
ax = fig.add_subplot(1,1,1)

# plot quivers
u1 = np.linspace(-3,3,20)
u2 = np.linspace(-3,3,20)

U1, U2 = np.meshgrid(u1, u2)
DU1, DU2 = system([U1, U2])
clrMap = (np.hypot(DU1, DU2))
clrMap[ clrMap==0 ] = 1
DU1 /= clrMap
DU2 /= clrMap

Q = ax.quiver(U1, U2, DU1, DU2, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{i}$')
ax.set_ylabel(f"$u_{j}$")
ax.grid()

g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A, valstep=0.05)
a0_constants_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'$a_{i}$ constant slider', valmin=0.1, valmax=1, valinit=a[i], valstep=0.01)
a1_constants_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'$a_{j}$ constant slider', valmin=0.1, valmax=1, valinit=a[j], valstep=0.01)
I0_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), f'$I_{i}$ slider', valmin=-15, valmax=15, valinit=I[i], valstep=0.01)
I0_slider = Slider(plt.axes([0.25, 0.35, 0.65, 0.03]), f'$I_{j}$ slider', valmin=-15, valmax=15, valinit=I[j], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.4, 0.65, 0.03]), '$s_{%s}$ slider' % (str(i)+str(j)), valmin=-10, valmax=10, valinit=s[0][1], valstep=0.1)

g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
a0_constants_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
s_slider.on_changed(update_plot)

plt.show()