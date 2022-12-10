from network_state import network
import numpy as np
import matplotlib.pyplot as plt

t_values = np.linspace(0, 360, 100000)

I = []
for t in t_values:
    stim = network.get_I(t)[0]
    I.append(stim)

for t in range(6):
    plt.axvline(x = 72*t, color = 'r')

plt.xlabel("$t$")
plt.ylabel("$I_i$")
plt.plot(t_values, I)
plt.grid()
plt.show()
