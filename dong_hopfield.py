from math import e as exp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# important constants
NUMBER_OF_NEURONS = 2
PLOT_SIZE = 4
NEURON_1_ID = 0
NEURON_2_ID = 1

# important variables
a_constants = [0.24, 0.63]

# mathematical functions
def sigmoid(x):
    return 1/(1+exp**(-x))

def deriv_neuron_state_wrt_time(neuron_state, # u
                                g_constant,
                                A_constant,
                                recent_correllation = [[0, 40]], # s
                                external_stimulus = [10, 1.23], # I,
                            ):
    term_1 = -neuron_state
    
    sum = 0
    connection_pointer = 0
    connection_strength=[sigmoid(item) for item in recent_correllation[0]] # T
    while connection_pointer <= NUMBER_OF_NEURONS-1:
        if connection_pointer != NEURON_1_ID:
            sum += connection_strength[NEURON_1_ID] * sigmoid(neuron_state)
        connection_pointer +=1
    term_2 = g_constant * sum

    term_3 = A_constant * external_stimulus[NEURON_1_ID]

    derivative = 1/a_constants[NEURON_1_ID] * (term_1 + term_2 + term_3)
    return derivative

def deriv_recent_correllation_wrt_time(
                                        neuron_1_state, # u_i
                                        neuron_2_state, # u_j
                                        recent_correllation, # s
                                        B_constants,
                                        H_constant
                                    ):
    term_1 = recent_correllation[NEURON_1_ID][NEURON_2_ID]

    term_2 = H_constant*sigmoid(neuron_1_state)*sigmoid(neuron_2_state)

    derivative = 1/B_constants[0][1](term_1 + term_2)

    return derivative

# necessary plotting functions
def determine_u_plot_data(g_constant, A_constant):
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_neuron_state_wrt_time(neuron_state=neuron_state, g_constant=g_constant, A_constant=A_constant)
            vector_hor_strength[i,j] = neuron_state
            vector_vert_strength[i,j] = derivative

def determine_s_plot_data(
                            neuron_1_state, # u_i
                            neuron_2_state, # u_j
                            neuron_id_1, # i
                            neuron_id_2, # j
                        ):
    pass

def update_u_plot(*args):
    determine_u_plot_data(g_constant=g_constant_slider.val,
                            A_constant=A_constant_slider.val
    )
    Q.set_UVC(vector_vert_strength, vector_hor_strength)
    fig.canvas.draw()

def update_a_constants(val):
    a_constants[NEURON_1_ID] = val
    update_u_plot()

# plotting preparation
hor_pos = np.linspace(-PLOT_SIZE,PLOT_SIZE,20)
vert_pos = np.linspace(-PLOT_SIZE,PLOT_SIZE,20)
HOR_pos, VERT_pos = np.meshgrid(hor_pos, vert_pos)
vector_vert_strength, vector_hor_strength = np.zeros(HOR_pos.shape), np.zeros(VERT_pos.shape)
VERT, HOR = HOR_pos.shape

# plotting window setup
fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(7)
fig.tight_layout(pad=5.0)
fig.subplots_adjust(bottom=0.5, hspace=0.5)

# subplot setup
ax = fig.subplots(2)
ax[0].set_xlim([-PLOT_SIZE, PLOT_SIZE])
ax[0].set_ylim([-PLOT_SIZE, PLOT_SIZE])
ax[0].set_xlabel('$u_i$')
ax[0].set_ylabel('$du_i/dt$')

ax[1].set_xlim([-PLOT_SIZE, PLOT_SIZE])
ax[1].set_ylim([-PLOT_SIZE, PLOT_SIZE])
ax[1].set_xlabel('$u_i$')
ax[1].set_ylabel('$du_i/dt$')

# create sliders
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=0, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=0, valstep=0.01)
a_constant_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), 'a constant slider', valmin=-0.5, valmax=0.5, valinit=0, valstep=0.01)

# slider updates
g_constant_slider.on_changed(update_u_plot)
A_constant_slider.on_changed(update_u_plot)
a_constant_slider.on_changed(update_a_constants)

# gather data
determine_u_plot_data(g_constant=g_constant_slider.val, 
                        A_constant=A_constant_slider.val,
                    )

# create plot
Q = ax[0].quiver(HOR_pos, VERT_pos, vector_vert_strength, vector_hor_strength)
plt.show()
