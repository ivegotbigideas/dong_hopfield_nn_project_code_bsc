from network_state import network
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 360, 100000)

I = []
for i in t:
    stim = network.get_I(i)[0]
    I.append(stim)

for i in range(6):
    plt.axvline(x = 72*i, color = 'r')

plt.plot(t, I)
plt.grid()
plt.show()
