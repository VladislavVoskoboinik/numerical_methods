import numpy as np
import math 
import matplotlib.pyplot as plt

t_0 = 2.
T = 10.
y_00 = 5
y_10 = 3
y_20 = 4

M_classic = 100
M = 250

eps = 0.01
p = 3
_t, tau = np.linspace(t_0, T, M_classic + 1, retstep = True) #retstep - возвращает шаг

u_0 = np.empty((M_classic + 1, 3))
u_DIRK1 = np.empty((M + 1, 3))

u_0[0] = [y_00, y_10, y_20]
u_DIRK1[0] = [y_00, y_10, y_20]


def f(u, x):
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 
    return f


def f_u(f,u, t, tau):
    f_u = (f(u, t + tau) - f(u, t))/tau
    return f_u



for m in range(0, M_classic): #ERK1
  u_0[m + 1] = u_0[m] + tau * f(u_0[m], _t[m])


t, tau = np.linspace(t_0, T, M + 1, retstep = True) #retstep - возвращает шаг
alpha = 0.5
for m in range(0, M): #DIRK1
  #w_1 = f(u_DIRK1[m], t[m] + tau/2, lambd)
  w_1= np.linalg.solve(np.eye(3) - alpha*tau*f_u(f, u_DIRK1[m], t[m], tau),f(u_DIRK1[m],t[m] + tau/2))
  u_DIRK1[m + 1] = u_DIRK1[m] + tau*w_1.real

plt.plot(_t, u_0[:,0], '-y')
plt.plot(t, u_DIRK1[:,0], '-g')

plt.xlim((2, 10)), plt.ylim(0, 70)
plt.show()