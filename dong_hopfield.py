from math import e as exp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# script constants
NUMBER_OF_NEURONS = 2
PLOT_SIZE = 40
NEURON_I_ID = 0
NEURON_J_ID = 1

# equation constant default values
a_constants = [0.24, 0.63]
g_constant = 0
A_constant = 0
H_constant = 0
B_constants = [[0, 1.8]]

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

def deriv_recent_correllation_wrt_time(recent_corr):
    term_1 = -recent_corr

    term_2 = H_constant*sigmoid(neuron_states[NEURON_I_ID])*sigmoid(neuron_states[NEURON_J_ID])

    derivative = (1/B_constants[NEURON_I_ID][NEURON_J_ID])*(term_1 + term_2)

    return derivative

# necessary plotting functions
def determine_u_plot_data():
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_neuron_state_wrt_time(neuron_state)
            vector_hor_strength[i,j] = neuron_state
            vector_vert_strength[i,j] = derivative

def determine_s_plot_data():
    for i in range(VERT):
        for j in range(HOR):
            recent_corr = HOR_pos[i,j]
            derivative = deriv_recent_correllation_wrt_time(recent_corr)
            vector_hor_strength[i,j] = recent_corr
            vector_vert_strength[i,j] = derivative

def update_u_plot(*args):
    global g_constant
    g_constant = g_constant_slider.val

    global A_constant
    A_constant = A_constant_slider.val

    global external_stimulus
    external_stimulus[NEURON_I_ID] = I_slider.val

    global a_constants
    a_constants[NEURON_I_ID] = a_constants_slider.val

    determine_u_plot_data()
    Q1.set_UVC(vector_vert_strength, vector_hor_strength)
    fig.canvas.draw()

def update_s_plot(*args):
    global H_constant
    H_constant = H_constant_slider.val

    determine_s_plot_data()
    Q2.set_UVC(vector_vert_strength, vector_hor_strength)
    fig2.canvas.draw()

def update_recent_correllation(val):
    recent_correllation[NEURON_I_ID][NEURON_J_ID] = val
    update_u_plot()

def update_B_constants(val):
    B_constants[NEURON_I_ID][NEURON_J_ID] = val
    update_s_plot()

def update_ui_values(val):
    neuron_states[NEURON_I_ID] = val
    update_s_plot()

def update_uj_values(val):
    neuron_states[NEURON_J_ID] = val
    update_s_plot()

# plot preparation
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

# u subplot setup
ax = fig.subplots(1)
ax.set_xlim([-PLOT_SIZE, PLOT_SIZE])
ax.set_ylim([-PLOT_SIZE, PLOT_SIZE])
ax.set_xlabel(f'$u_{NEURON_I_ID}$')
ax.set_ylabel(f'$du_{NEURON_I_ID}/dt$')

# create u plot sliders
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=g_constant, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=A_constant, valstep=0.05)
a_constants_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'a_{NEURON_I_ID} constant slider', valmin=0.1, valmax=0.5, valinit=a_constants[NEURON_I_ID], valstep=0.01)
s_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f's_{NEURON_I_ID}{NEURON_J_ID} slider', valmin=0, valmax=5, valinit=recent_correllation[NEURON_I_ID][NEURON_J_ID], valstep=0.05)
I_slider = Slider(plt.axes([0.25, 0.3, 0.65, 0.03]), 'I constant slider', valmin=-15, valmax=15, valinit=external_stimulus[NEURON_I_ID], valstep=0.01)

# u plot slider updates
g_constant_slider.on_changed(update_u_plot)
A_constant_slider.on_changed(update_u_plot)
a_constants_slider.on_changed(update_u_plot)
s_slider.on_changed(update_recent_correllation)
I_slider.on_changed(update_u_plot)

# gather u plot data
determine_u_plot_data()

# create u plot
Q1 = ax.quiver(HOR_pos, VERT_pos, vector_vert_strength, vector_hor_strength)

# s plot window setup
fig2 = plt.figure()
fig2.set_figwidth(7)
fig2.set_figheight(7)
fig2.tight_layout(pad=5.0)
fig2.subplots_adjust(bottom=0.5, hspace=0.5)

# s subplot setup
ax2 = fig2.subplots(1)
ax2.set_xlim([-PLOT_SIZE, PLOT_SIZE])
ax2.set_ylim([-PLOT_SIZE, PLOT_SIZE])
ax2.set_xlabel(f'$s{NEURON_I_ID}{NEURON_J_ID}$')
ax2.set_ylabel(f'$ds{NEURON_I_ID}{NEURON_J_ID}/dt$')

# create u plot sliders
H_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'H constant slider', valmin=-15, valmax=15, valinit=H_constant, valstep=0.01)
B_constant_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), f'B_{NEURON_I_ID}{NEURON_J_ID} slider', valmin=-1, valmax=1, valinit=B_constants[NEURON_I_ID][NEURON_J_ID], valstep=0.05)
u_i_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), f'u_{NEURON_I_ID} constant slider', valmin=-3, valmax=3, valinit=neuron_states[NEURON_I_ID], valstep=0.01)
u_j_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), f'u_{NEURON_J_ID} constant slider', valmin=-3, valmax=3, valinit=neuron_states[NEURON_J_ID], valstep=0.01)

# s plot slider updates
H_constant_slider.on_changed(update_s_plot)
B_constant_slider.on_changed(update_B_constants)
u_i_slider.on_changed(update_ui_values)
u_j_slider.on_changed(update_uj_values)

# gather s plot data
determine_s_plot_data()

# create s plot
Q2 = ax2.quiver(HOR_pos, VERT_pos, vector_vert_strength, vector_hor_strength)

# show plots
plt.show()
