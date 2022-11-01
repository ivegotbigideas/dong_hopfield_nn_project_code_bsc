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
s = [[0, 0.4]]
I = [10, 1.23]
a = [0.24, 0.63]

# mathematical constants
g = 0.5
A = 2
H = 1.3
B = [[0, 1.8]]

# mathematical functions
def sigmoid(x):
    return 1/(1+exp**(-x))

def dudt(u):
    term_1 = -u[i]

    sum = 0
    connection_pointer = 0
    connection_strength=[sigmoid(item) for item in s[i]] # T
    while connection_pointer <= NUMBER_OF_NEURONS-1:
        if connection_pointer != i:
            sum += connection_strength[connection_pointer] * sigmoid(u[i])
        connection_pointer +=1
    term_2 = g*sum

    term_3 = A * I[i]

    derivative = 1/a[i] * (term_1 + term_2 + term_3)
    return derivative

def dsdt(s, u):
    term_1 = -s[i][j]

    term_2 = H*sigmoid(u[i])*sigmoid(u[i])

    derivative = (1/B[i][j])*(term_1 + term_2)

    return derivative

def system(u, s):
    return np.array(dudt(u), dsdt(u,s))

# plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)

u = np.linspace(0,2,20)
s = np.linspace(0, 2, 20)

U, S = np.meshgrid(u, s)
DU, DS = system(U, S)
M = (np.hypot(DU, DS))
M[ M==0 ] = 1
DU /= M
DS /= M

ax.quiver(U, S, DU, DS, M, pivot='mid')
ax.legend()
ax.grid()

plt.show()