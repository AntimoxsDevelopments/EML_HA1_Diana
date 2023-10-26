# Abgabegruppe: Diana Kanewski, Irini Metsios, Timo Heitmann
import numpy as np
import matplotlib.pyplot as plt


def least_squares(a):
    x, y = np.stack(a, axis=1)
    x_av = np.average(x)
    y_av = np.average(y)
    w = np.sum((y - y_av) * x) / np.sum((x - x_av) * x)
    b = y_av - w * x_av
    return w, b


data0 = np.load('dataset0.npy')
data1 = np.load('dataset1.npy')
data2 = np.load('dataset2.npy')

x_range = np.linspace(0, 3, 100)

fig, ax = plt.subplots()
ax.scatter(*np.stack(data0, axis=1), c='#D32F2F', label='dataset0', s=5)
ax.scatter(*np.stack(data1, axis=1), c='#F57C00', label='dataset1', s=5)
ax.scatter(*np.stack(data2, axis=1), c='#FFCC00', label='dataset2', s=5)

w0, b0 = least_squares(data0)
w1, b1 = least_squares(data1)
w2, b2 = least_squares(data2)

plt.plot(x_range, w0 * x_range + b0, c='#689F38', label='linear regression of dataset0')
plt.plot(x_range, w1 * x_range + b1, c='#1976D2', label='linear regression of dataset1')
plt.plot(x_range, w2 * x_range + b2, c='#6E40BF', label='linear regression of dataset2')

plt.title('Visualization of datasets')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.xlim(0, 3)
plt.legend()
plt.show()
