from math import e as exp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

number_of_neurons = 2

def sigmoid(x):
    return 1/(1+exp**(-x))

# Initial conditions
recent_correllation = [[0, 0.4]] # s_01 = s_10
connection_strength = [sigmoid(item) for item in recent_correllation[0]] # T_01 = T_10
external_stimulus = [0.53, 1.23]
a_constants = [0.24, 0.63]
B_constants = [[0, 1.3]]
g_constant = 150
H_constant = 0.87
A_constant = 0.3

# function_definitions
def deriv_neuron_state_wrt_time(neuron_id, neuron_state):
    term_1 = -neuron_state
    
    sum = 0
    connection_pointer = 0
    while connection_pointer <= number_of_neurons-1:
        if connection_pointer != neuron_id:
            sum += connection_strength[neuron_id] * sigmoid(neuron_state)
        connection_pointer +=1
    print(g_constant)
    term_2 = g_constant * sum

    term_3 = A_constant * external_stimulus[neuron_id]

    derivative = 1/a_constants[neuron_id] * (term_1 + term_2 + term_3)
    return derivative

def deriv_recent_correllation_wrt_time(neuron_id_1, neuron_id_2, neuron_1_state, neuron_2_state):
    term_1 = recent_correllation[neuron_id_1][neuron_id_2]

    term_2 = H_constant*sigmoid(neuron_1_state)*sigmoid(neuron_2_state)

    derivative = 1/B_constants[0][1](term_1 + term_2)

    return derivative

# plotting preparation
plot_size = 400
hor_pos = np.linspace(0,plot_size,20)
vert_pos = np.linspace(0,plot_size,20)
HOR_pos, VERT_pos = np.meshgrid(hor_pos, vert_pos)
VERT_strength, HOR_strength = np.zeros(HOR_pos.shape), np.zeros(VERT_pos.shape)
VERT, HOR = HOR_pos.shape

def determine_plot_data():
    for i in range(VERT):
        for j in range(HOR):
            neuron_state = HOR_pos[i,j]
            derivative = deriv_neuron_state_wrt_time(neuron_id=0, neuron_state=neuron_state)
            HOR_strength[i,j] = neuron_state
            VERT_strength[i,j] = derivative

def update_plot(slider_val):
    global g_constant
    g_constant = slider_val
    determine_plot_data()
    Q.set_UVC(VERT_strength, HOR_strength)
    fig.canvas.draw()

# plotting
determine_plot_data()
fig = plt.figure()
ax = fig.subplots()
plt.subplots_adjust(bottom=0.25)
plt.xlim([0, plot_size])
plt.ylim([0, plot_size])
ax.set_xlabel('$u_0$')
ax.set_ylabel('$du_0/dt$')

ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
g_constant_slider = Slider(ax_slide, 'g constant slider', valmin=0.1, valmax=150, valinit = 6, valstep=5)
g_constant_slider.on_changed(update_plot)

Q = ax.quiver(HOR_pos, VERT_pos, VERT_strength, HOR_strength)
plt.show()
