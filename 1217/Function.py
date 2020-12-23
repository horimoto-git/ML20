import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class function1():
    def __init__(self):
        """
        self.boundaries is the range of x and y axis
        """
        self.boundaries = np.array([-5.12, 5.12])

    def f(self, x, y):
        """
        Function (Z) value
        """
        t1 = 10 * 2
        t2 = x**2+y**2
        t3 = - 10 * np.cos(2 * np.pi * x)+10 * np.cos(2 * np.pi * y)
        return t1 + t2 + t3

class function2():
    def __init__(self):
        """
        self.boundaries is the range of x and y axis
        """
        self.boundaries = np.array([-5, 5])

    def f(self, x, y):
        """
        Function (Z) value
        """
        t1 = 100 * (y - x**2) ** 2
        t2 = (x - 1) ** 2
        return t1+t2

N  = 1000
min = -5.12
max  = 5.12
x = np.linspace( min, max, N)   # linspace(min, max, N) で範囲 min から max を N 分割
y = np.linspace( min, max, N)
x, y = np.meshgrid(x, y) # plot_surface関数に渡すデータは，2次元配列

z = function1.f(1, x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig.savefig("function1")
plt.show()

N  = 1000
min = -5
max  = 5
x = np.linspace( min, max, N)   # linspace(min, max, N) で範囲 min から max を N 分割
y = np.linspace( min, max, N)
x, y = np.meshgrid(x, y) # plot_surface関数に渡すデータは，2次元配列

z = function2.f(0, x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig.savefig("function2")
plt.show()
