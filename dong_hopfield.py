from math import e as exp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_NEURONS = 2

def sigmoid(x):
    return 1/(1+exp**(-x))

# function_definitions
def deriv_neuron_state_wrt_time(neuron_id, # i
                                neuron_state, # u
                                g_constant,
                                A_constant,
                                recent_correllation = [[0, 40]], # s
                                external_stimulus = [10, 1.23], # I,
                                a_constants = [0.24, 0.63],
                                ):
    term_1 = -neuron_state
    
    sum = 0
    connection_pointer = 0
    connection_strength=[sigmoid(item) for item in recent_correllation[0]] # T
    while connection_pointer <= NUMBER_OF_NEURONS-1:
        if connection_pointer != neuron_id:
            sum += connection_strength[neuron_id] * sigmoid(neuron_state)
        connection_pointer +=1
    term_2 = g_constant * sum

    term_3 = A_constant * external_stimulus[neuron_id]

    derivative = 1/a_constants[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def deriv_recent_correllation_wrt_time(neuron_id_1, # i
                                        neuron_id_2, # j
                                        neuron_1_state, # u_i
                                        neuron_2_state, # u_j
                                        recent_correllation = [[0, 0.4]], # s
                                        B_constants = [[0, 1.3]],
                                        H_constant = 0.87
                                        ):
    term_1 = recent_correllation[neuron_id_1][neuron_id_2]

    term_2 = H_constant*sigmoid(neuron_1_state)*sigmoid(neuron_2_state)

    derivative = 1/B_constants[0][1](term_1 + term_2)

    return derivative

# plotting preparation
plot_size = 4
hor_pos = np.linspace(-plot_size,plot_size,20)
vert_pos = np.linspace(-plot_size,plot_size,20)
HOR_pos, VERT_pos = np.meshgrid(hor_pos, vert_pos)
vector_vert_strength, vector_hor_strength = np.zeros(HOR_pos.shape), np.zeros(VERT_pos.shape)
VERT, HOR = HOR_pos.shape

# plotting setup
fig = plt.figure()
ax = fig.subplots()
ax.set_xlim([-plot_size, plot_size])
ax.set_ylim([-plot_size, plot_size])
ax.set_xlabel('$u_0$')
ax.set_ylabel('$du_0/dt$')

# create sliders
fig.subplots_adjust(bottom=0.5)
g_constant_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'g constant slider', valmin=-15, valmax=15, valinit=0, valstep=0.01)
A_constant_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'A constant slider', valmin=-1.5, valmax=1.5, valinit=0, valstep=0.01)

# necessary plotting functions
def determine_u_plot_data(neuron_id=0, g_constant=g_constant_slider.val, A_constant=A_constant_slider.val):
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_neuron_state_wrt_time(neuron_id=neuron_id, neuron_state=neuron_state, g_constant=g_constant, A_constant=A_constant)
            vector_hor_strength[i,j] = neuron_state
            vector_vert_strength[i,j] = derivative

def update_u_plot(slider_val):
    determine_u_plot_data(g_constant=g_constant_slider.val,
                        A_constant=A_constant_slider.val
    )
    Q.set_UVC(vector_vert_strength, vector_hor_strength)
    fig.canvas.draw()

# slider updates
g_constant_slider.on_changed(update_u_plot)
A_constant_slider.on_changed(update_u_plot)

# physical plotting
determine_u_plot_data(neuron_id=0)
Q = ax.quiver(HOR_pos, VERT_pos, vector_vert_strength, vector_hor_strength)
plt.show()
