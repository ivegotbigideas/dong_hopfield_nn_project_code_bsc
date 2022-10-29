from math import e as exp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# script constants
NUMBER_OF_NEURONS = 2
PLOT_SIZE = 4
NEURON_I_ID = 0
NEURON_J_ID = 1

# equation constant default values
a_constants = [0.24, 0.63]
g_constant = 0
A_constant = 0

# network state
recent_correllation = [[0, 0.4]] # s
external_stimulus = [10, 1.23] # I
neuron_states = [2.5, 3.7]

# mathematical functions
def sigmoid(x):
    return 1/(1+exp**(-x))

def deriv_neuron_state_wrt_time(neuron_state):
    term_1 = -neuron_state
    
    sum = 0
    connection_pointer = 0
    connection_strength=[sigmoid(item) for item in recent_correllation[NEURON_I_ID]] # T
    while connection_pointer <= NUMBER_OF_NEURONS-1:
        if connection_pointer != NEURON_I_ID:
            sum += connection_strength[NEURON_J_ID] * sigmoid(neuron_state)
        connection_pointer +=1
    term_2 = g_constant * sum

    term_3 = A_constant * external_stimulus[NEURON_I_ID]

    derivative = 1/a_constants[NEURON_I_ID] * (term_1 + term_2 + term_3)
    return derivative

def deriv_recent_correllation_wrt_time(
                                        neuron_i_state, # u_i
                                        neuron_j_state, # u_j
                                        recent_correllation, # s_ij
                                        B_constants,
                                        H_constant
                                    ):
    term_1 = recent_correllation[NEURON_I_ID][NEURON_J_ID]

    term_2 = H_constant*sigmoid(neuron_i_state)*sigmoid(neuron_j_state)

    derivative = (1/B_constants[NEURON_I_ID][NEURON_J_ID])*(term_1 + term_2)

    return derivative

# necessary plotting functions
def determine_u_plot_data():
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_neuron_state_wrt_time(neuron_state=neuron_state)
            vector_hor_strength[i,j] = neuron_state
            vector_vert_strength[i,j] = derivative

def determine_s_plot_data(
                            neuron_1_state, # u_i
                            neuron_2_state, # u_j
                            neuron_id_1, # i
                            neuron_id_2, # j
                        ):
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_recent_correllation_wrt_time()
            vector_hor_strength[i,j] = neuron_state
            vector_vert_strength[i,j] = derivative

def update_u_plot(*args):
    global g_constant
    g_constant = g_constant_slider.val

    global A_constant
    A_constant = A_constant_slider.val

    determine_u_plot_data()
    Q1.set_UVC(vector_vert_strength, vector_hor_strength)
    fig.canvas.draw()

def update_a_constants(val):
    a_constants[NEURON_I_ID] = val
    update_u_plot()

def update_recent_correllation(val):
    recent_correllation[NEURON_I_ID][NEURON_J_ID] = val
    update_u_plot()

# u plot preparation
hor_pos = np.linspace(-PLOT_SIZE,PLOT_SIZE,20)
vert_pos = np.linspace(-PLOT_SIZE,PLOT_SIZE,20)
HOR_pos, VERT_pos = np.meshgrid(hor_pos, vert_pos)
vector_vert_strength, vector_hor_strength = np.zeros(HOR_pos.shape), np.zeros(VERT_pos.shape)
VERT, HOR = HOR_pos.shape

# u plot window setup
fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(7)
fig.tight_layout(pad=5.0)
fig.subplots_adjust(bottom=0.5, hspace=0.5)

# subplot setup
ax = fig.subplots(1)
ax.set_xlim([-PLOT_SIZE, PLOT_SIZE])
ax.set_ylim([-PLOT_SIZE, PLOT_SIZE])
ax.set_xlabel(f'$u_{NEURON_I_ID}$')
ax.set_ylabel(f'$du_{NEURON_I_ID}/dt$')

# create u plot sliders
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g_constant, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A_constant, valstep=0.01)
a_constant_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'a_{NEURON_I_ID} constant slider', valmin=0.1, valmax=0.5, valinit=a_constants[NEURON_I_ID], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f's_{NEURON_I_ID}{NEURON_J_ID} slider', valmin=0, valmax=5, valinit=recent_correllation[NEURON_I_ID][NEURON_J_ID], valstep=0.05)

# u plot slider updates
g_constant_slider.on_changed(update_u_plot)
A_constant_slider.on_changed(update_u_plot)
a_constant_slider.on_changed(update_a_constants)
s_slider.on_changed(update_recent_correllation)

# gather u plot data
determine_u_plot_data()

# create u plot
Q1 = ax.quiver(HOR_pos, VERT_pos, vector_vert_strength, vector_hor_strength)

# show plots
plt.show()
