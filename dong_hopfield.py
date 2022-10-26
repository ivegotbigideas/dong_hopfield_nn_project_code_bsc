from math import e as exp
import matplotlib.pyplot as plt
import numpy as np

number_of_neurons = 2

def sigmoid(x): # 
    return 1/(1+exp**(-x))

# Initial conditions
recent_correllation = [0.4] # s_01 = s_10
connection_strength = [sigmoid(item) for item in recent_correllation] # T_01 = T_10
external_stimulus = 0.53
a_constants = [0.24, 0.63]
g_constant = 0.15
A_constant = 0.3

# function_definitions
def deriv_neuron_state_wrt_time(neuron_id, neuron_state):
    term_1 = -neuron_state
    
    sum = 0
    connection_pointer = 0
    while connection_pointer < number_of_neurons-1:
        if connection_pointer != neuron_id:
            sum += connection_strength[connection_pointer] * sigmoid(neuron_state)
        connection_pointer +=1
    term_2 = g_constant * sum

    term_3 = A_constant * external_stimulus

    derivative = 1/a_constants[neuron_id] * (term_1 + term_2 + term_3)

    return derivative

# plotting
plot_size = 100
hor_pos = np.linspace(-plot_size,plot_size,20)
vert_pos = np.linspace(-plot_size,plot_size,20)
HOR_pos, VERT_pos = np.meshgrid(hor_pos, vert_pos)

VERT_strength, HOR_strength = np.zeros(HOR_pos.shape), np.zeros(VERT_pos.shape)

VERT, HOR = HOR_pos.shape
for i in range(VERT):
    for j in range(HOR):
        neuron_state = HOR_pos[i,j]
        derivative = deriv_neuron_state_wrt_time(neuron_id=0, neuron_state=neuron_state)
        HOR_strength[i,j] = neuron_state
        VERT_strength[i,j] = derivative

plt.xlabel('$u_0$')
plt.ylabel('$du_0/dt$')
plt.xlim([-plot_size, plot_size])
plt.ylim([-plot_size, plot_size])
plt.quiver(HOR_pos, VERT_pos, VERT_strength, HOR_strength)
plt.show()