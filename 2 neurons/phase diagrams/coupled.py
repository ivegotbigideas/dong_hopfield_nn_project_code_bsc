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
I = [2.5, 10]
a = [0.24, 0.63]

# mathematical constants
g = 0.5
A = 2
H = 1.3
B = [[0, 1.8]]

# mathematical functions
def sigmoid(x):
    return 1/(1+exp**(-x))

def dudt(u, s):
    term_1 = -u

    sum = 0
    connection_pointer = 0
    connection_strength=sigmoid(s) # T
    while connection_pointer < NUMBER_OF_NEURONS:
        if connection_pointer != i:
            sum += connection_strength * sigmoid(u)
        connection_pointer +=1
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

# find fixed points
def find_fixed_points(rng):
    fixed_points = []
    for u in range(rng):
        for s in range(rng):
            if ((dudt(u, s) == 0) and (dsdt(u, s) == 0)):
                fixed_points.append((u,s))
    return fixed_points

fixed_points = find_fixed_points(50)

# setup plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)

# plot fixed points
for point in fixed_points:
    print(point)
    ax.plot(point[0],point[1],"red", marker = "o", markersize = 10.0)

# plot quivers
u = np.linspace(-50,50,20)
s = np.linspace(-50,50,20)

U, S = np.meshgrid(u, s)
DU, DS = system(U, S)
clrMap = (np.hypot(DU, DS))
clrMap[ clrMap==0 ] = 1
DU /= clrMap
DS /= clrMap

ax.quiver(U, S, DU, DS, clrMap, pivot='mid')
ax.set_xlabel(f'$u_{i}$')
ax.set_ylabel("$s_{%s}$" % (str(i)+str(j)))
ax.grid()

plt.show()