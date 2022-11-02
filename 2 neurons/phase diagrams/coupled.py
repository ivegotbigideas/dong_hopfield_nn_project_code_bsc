from math import e as exp
from matplotlib.widgets import Slider
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

# script constants
NUMBER_OF_NEURONS = 2
i = 1
j = 0

# network state
I = [2.5, 10]

# equation constants
g = 5
a = [0.24, 0.63]
A = 2
H = 1.3
B = np.array([ [0, 1.8], [1.8, 0] ])

# mathematical functions
def sigmoid(x):
    return 1/(1+exp**(-x))

def dudt(u, s):
    term_1 = -u

    connection_strength=sigmoid(s) # T
    sum = connection_strength * sigmoid(u)
    term_2 = g*sum

    term_3 = A * I[i]

    derivative = 1/a[i] * (term_1 + term_2 + term_3)
    return derivative

def dsdt(u, s):
    term_1 = -s

    term_2 = H*sigmoid(u)*sigmoid(u)

    derivative = (1/B[i][j])*(term_1 + term_2)

    return derivative

def system(u, s):
    return np.array([dudt(u, s), dsdt(u, s)])

# plotting functions
def update_plot(*args):
    global g
    g = g_constant_slider.val

    global A
    A = A_constant_slider.val

    global H
    H = H_constant_slider.val

    global a
    a[i] = a_constants_slider.val

    global B
    B[i][j] = B_constant_slider.val

    global I
    I[i] = I_slider.val

    DU, DS = system(U, S)
    clrMap = (np.hypot(DU, DS))
    clrMap[ clrMap==0 ] = 1
    DU /= clrMap
    DS /= clrMap

    Q.set_UVC(DU, DS)
    fig.canvas.draw()


# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
fig.subplots_adjust(bottom=0.5, hspace=0.5)
ax = fig.add_subplot(1,1,1)

# plot quivers
u = np.linspace(-50,50,20)
s = np.linspace(-50,50,20)

U, S = np.meshgrid(u, s)
DU, DS = system(U, S)
clrMap = (np.hypot(DU, DS))
clrMap[ clrMap==0 ] = 1
DU /= clrMap
DS /= clrMap

Q = ax.quiver(U, S, DU, DS, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{i}$')
ax.set_ylabel("$s_{%s}$" % (str(i)+str(j)))
ax.grid()

g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A, valstep=0.05)
H_constant_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'H constant slider', valmin=-15, valmax=15, valinit=H, valstep=0.01)
a_constants_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'a_{i} constant slider', valmin=0.1, valmax=0.5, valinit=a[i], valstep=0.01)
B_constant_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), f'B_{i}{j} slider', valmin=-1, valmax=1, valinit=B[i][j], valstep=0.05)
I_slider = Slider(plt.axes([0.25, 0.35, 0.65, 0.03]), 'I constant slider', valmin=-15, valmax=15, valinit=I[i], valstep=0.01)

g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
H_constant_slider.on_changed(update_plot)
a_constants_slider.on_changed(update_plot)
B_constant_slider.on_changed(update_plot)
I_slider.on_changed(update_plot)

plt.show()