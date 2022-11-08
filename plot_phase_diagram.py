from math import e as exp
from matplotlib.widgets import Slider
from mathematical_functions import focal_neurons, two_dim_system, find_fixed_points_of_2D_system
from network_state import I, s, g, a, A
import matplotlib.pyplot as plt
import numpy as np

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
    I[focal_neurons[1]] = I1_slider.val

    global s
    s[focal_neurons[0]][focal_neurons[1]] = s_slider.val
    s[focal_neurons[1]][focal_neurons[0]] = s_slider.val

    DU0, DU1 = two_dim_system([U0, U1], I, s, g, a, A)
    clrMap = (np.hypot(DU0, DU1))
    clrMap[ clrMap==0 ] = 1
    DU0 /= clrMap
    DU1 /= clrMap

    fp = find_fixed_points_of_2D_system(I, s, g, a, A)
    for marker in range(len(C)):
        C[marker][0].set_data(fp[marker])

    Q.set_UVC(DU0, DU1)
    fig.canvas.draw()

# setup plot
fig = plt.figure(figsize=(8,6))
fig.tight_layout(pad=5.0)
ax = fig.add_subplot(1,1,1)

# prepare data
u0 = np.linspace(-6,6,20)
u1 = np.linspace(-6,6,20)

U0, U1 = np.meshgrid(u0, u1)
DU0, DU1 = two_dim_system([U0, U1], I, s, g, a, A)
clrMap = (np.hypot(DU0, DU1))
clrMap[ clrMap==0 ] = 1
DU0 /= clrMap
DU1 /= clrMap

fp = find_fixed_points_of_2D_system(I, s, g, a, A)
C = []
for point in fp:
    C.append(ax.plot(point[0], point[1],"red", marker = "x", markersize = 7.0))

# plot quivers
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
s_slider = Slider(plt.axes([0.25, 0.4, 0.65, 0.03]), '$s_{%s}$ slider' % (str(focal_neurons[0])+str(focal_neurons[1])), valmin=0, valmax=30, valinit=s[focal_neurons[0]][focal_neurons[1]], valstep=0.1)

g_constant_slider.on_changed(update_plot)
A_constant_slider.on_changed(update_plot)
a0_constants_slider.on_changed(update_plot)
I0_slider.on_changed(update_plot)
I1_slider.on_changed(update_plot)
s_slider.on_changed(update_plot)

# display
plt.show()