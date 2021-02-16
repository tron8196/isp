import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
x_axis = np.linspace(-5,5, 100)
y_axis = (3*x_axis + 1)/2

print(y_axis)
plt.grid(True, which='both')
plt.plot(x_axis, y_axis)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.ylabel('some numbers')
y_axis_updated = tf.math.sigmoid(y_axis)

plt.plot(x_axis, y_axis_updated)
plt.show()  
