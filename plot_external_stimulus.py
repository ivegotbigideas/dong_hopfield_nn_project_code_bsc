from network_state import network
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

t_values = np.linspace(0, 360, 100000)

I = []
for t in t_values:
    stim = network.get_I(t)[0]
    I.append(stim)

for t in range(6):
    plt.axvline(x = 72*t, color = 'r', linestyle="dashed")

plt.xlim(0, 288)
plt.xlabel("$t$", fontsize=17)
plt.ylabel("$I_0$", fontsize=17)
plt.plot(t_values, I)
plt.xticks(np.arange(0, 360, 24.0))
plt.grid()
plt.show()
