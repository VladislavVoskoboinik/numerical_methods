import numpy as np
import matplotlib.pyplot as plt
import math 


t_0 = 0.
T = 100
x_0 = 0
y_0 = 0
v_0 = 150
alpha = np.pi / 4
g = 9.81
k = 10
mass = 500
u_0 = 1000
lambd = 0.5

L = 1000
J = 10000
dl = 30
''''
t = np.empty(M+1)
for m in range(0, M+1):
    t[m] = t_0 + tau*m 
'''''
u = np.empty((J, 2))
#u[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]
u[0] = [u_0, t_0]
def f(u):
    f = np.empty(2)
    f[0] = -lambd * u[0]
    f[1] = 1
    f = f / np.sqrt(f[0] ** 2 + 1)
    return f

j = 0
while u[j, 1] < T:
    u[j + 1] = u[j] + dl * f(u[j])
    j += 1

print(j)

'''''
for m in range(0, M):
    u[m + 1] = u[m] + tau * f(u[m])
'''''


plt.plot(u[:j+1, 1], u[:j+1, 0], '-or')
plt.plot(u[:j+1, 1], 0*u[:j+1, 0], '-xb')
plt.xlim((0, T)), plt.ylim((0, u_0))
plt.show()
