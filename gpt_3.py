import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

x = np.linspace(0, 2*np.pi, 200)
line, = ax.plot(x, np.sin(x))

def update(frame):
    line.set_ydata(np.sin(x + 0.1 * frame))
    return line,

ani = FuncAnimation(fig, update, frames=100, blit=True)
plt.show()