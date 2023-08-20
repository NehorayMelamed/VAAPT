import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to update the graph
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    line.set_data(xdata, ydata)
    return line,

# Initialize the graph
fig, ax = plt.subplots()
xdata, ydata = [], []
line, = plt.plot([], [], 'r-')
plt.xlim(0, 2 * np.pi)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('sin(x)')

# Create an animation with the update function
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), interval=100, blit=True)

# Display the graph
plt.show()
