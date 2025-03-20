import math
import numpy as np
import matplotlib.pyplot as plt

# differential equation: y'' = 2*(y-1) * ctg(x), y(0) = 3

# Параметры
x_0 = np.pi/2         # Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1         # Начальное значение w (y'')
X = 10
M = 50   # Конечное значение x
x_grid, tau =  np.linspace(x_0, X, M+1, retstep=True)

u = np.empty((M+1, 3))

u[0] = [y_0, v_0, w_0]

def f(u, x):
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f

for m in range(0, M):
    u[m+1] = u[m] + tau * f(u[m], x_grid[m])

# Аналитическое решение
y_analytical = 0.5 * x_grid**2 - np.pi * x_grid * 0.5 + 1 + (np.pi ** 2) / 8

# Построение графика
plt.plot(x_grid, u[:, 0], label='Численное решение', color = 'red')
plt.plot(x_grid, y_analytical, label='Аналитическое решение', linestyle = "-", color = 'green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Схема Эйлера')
plt.legend()
plt.grid(True)
plt.show()
