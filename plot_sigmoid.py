import matplotlib.pyplot as plt
import numpy as np
from mathematical_functions import sigmoid

x = np.linspace(-5, 5, 4000)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlabel("x")
ax.set_ylabel("\u03C3" + "(x)")
ax.yaxis.set_label_coords(0.52, 1.07)
ax.xaxis.set_label_coords(1.05, 0.52)

plt.plot(x, sigmoid(x))
plt.show()