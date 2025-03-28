import numpy as np
import matplotlib.pyplot as plt
import math 


x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 100     #Конец отрезка
M = 100000
tau = 0.2


L = 5000
J = 10000
dl = 30
''''
t = np.empty(M+1)
for m in range(0, M+1):
    t[m] = t_0 + tau*m 
'''''
u = np.empty((J, 4))
#u[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]
u[0] = [y_0, v_0, w_0, x_0]

def f(u, x):
    f = np.empty(4)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x)
    f[3] = 1
    f = f/ np.sqrt(f[0]**2 + f[1] ** 2 + f[2] ** 2 + 1)
    return f

j = 0
while u[j, 3] < X:
    u[j + 1] = u[j] + dl * f(u[j], u[j, 3])
    j += 1

print(j)


'''''
for m in range(0, M):
    u[m + 1] = u[m] + tau * f(u[m])
'''''


x_analytical = np.linspace(x_0, X, M*16)
y_analytical = 0.5 * x_analytical**2 - np.pi * x_analytical * 0.5 + 1 + (np.pi ** 2) / 8



plt.plot(u[:j+1, 3], u[:j+1, 0], '-or')
plt.plot(u[:j+1, 3], 0*u[:j+1, 0], '-xb')

plt.xlim((x_0, X)), plt.ylim((0, 10000))
plt.plot(x_analytical, y_analytical, label='Аналитическое решение', linestyle = "--", color = 'green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение уравнения третьего порядка')
plt.legend()
plt.grid(True)
plt.show()

