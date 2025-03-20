import numpy as np
import math 


t_0 = 0.
T = 7.
x_0 = 0
y_0 = 0
v_0 = 150
alpha = np.pi / 4
g = 9.81
k = 10
mass = 500

M = 50

tau =  (T - t_0) / M

''''
t = np.empty(M+1)
for m in range(0, M+1):
    t[m] = t_0 + tau*m 
'''''

t, tau = np.linspace(t_0, T, M+1, retstep = True) #retstep - возвращает шаг

u = np.empty((M + 1, 4))
u[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]

def f(u, t):
    f = np.empty(4)
    f[0] = u[2]
    f[1] = u[3]
    f[2] = -k/mass* math.sqrt(u[2] ** 2 + u[3]**2) * u[2]
    f[3] = -g -k/mass * math.sqrt(u[2] ** 2 + u[3] ** 2) * u[3]
    return f

for m in range(0, M):
    u[m + 1] = u[m] + tau * f(u[m], t[m])

import matplotlib.pyplot as plt

plt.plot(u[:,0], u[:,1], '-or')
plt.xlim((0, 1.62*80)), plt.ylim(0, 80)
plt.show()
