import numpy as np
import math 
import matplotlib.pyplot as plt

t_0 = 0.
T = 7.
x_0 = 0
y_0 = 0
v_0 = 150
alpha = np.pi / 4
g = 9.81
k = 10
mass = 500

M = 500 #max num of steps

tau =  0.14
eps = 1
''''
t = np.empty(M+1)
for m in range(0, M+1):
    t[m] = t_0 + tau*m 
'''''

t, tau = np.linspace(t_0, T, M+1, retstep = True) #retstep - возвращает шаг

u_ERK1 = np.empty((M + 1, 4))
u_ERK2 = np.empty((M + 1, 4))
u_ERK3 = np.empty((M + 1, 4))
u_ERK4 = np.empty((M + 1, 4))
u_ERK1[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]
u_ERK2[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]
u_ERK3[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]
u_ERK4[0] = [x_0, y_0, v_0*np.cos(alpha), v_0*np.sin(alpha)]


def f(u, t):
    f = np.empty(4)
    f[0] = u[2]
    f[1] = u[3]
    f[2] = -k/mass* math.sqrt(u[2] ** 2 + u[3]**2) * u[2]
    f[3] = -g -k/mass * math.sqrt(u[2] ** 2 + u[3] ** 2) * u[3]
    return f

'''''
#ERK_1
for m in range(0, M):
    u_ERK1[m + 1] = u_ERK1[m] + tau * f(u_ERK1[m], t[m])



#ERK_2
for m in range(0, M):
    w_1 = f(u_ERK2[m], t[m])
    w_2 = f(u_ERK2[m] + 2/3 * tau * w_1, t[m] + 2/3 * tau)
    u_ERK2[m + 1] = u_ERK2[m] + tau * (1/4 * w_1 + 3/4 * w_2)


'''''
#ERK_3
m = 0
p = 3
while t[m] < T:
    w_1 = f(u_ERK3[m], t[m])
    w_2 = f(u_ERK3[m] + 1/2 * tau * w_1, t[m] + 1/2 * tau)
    w_3 = f(u_ERK3[m] + 3/4 * tau * w_2, t[m] + 3/4 * tau)
    u_ERK3[m + 1] = u_ERK3[m] + tau * (2/9 * w_1 + 3/9 * w_2 + 4/9 * w_3)
    u_emb = u_ERK3[m] + tau * w_2
    tau = (eps * tau**p / (T - t_0) / (sum(u_ERK3[m + 1] - u_emb)**2)) ** (1/(p-1))
    u_ERK3[m + 1] = u_ERK3[m] + tau * (2/9 * w_1 + 3/9 * w_2 + 4/9 * w_3)
    t[m + 1] = t[m] + tau
    m += 1


'''''
#ERK_4
for m in range(0, M): #ERK4
    w1 = f(u_ERK4[m], t[m])
    w2 = f(u_ERK4[m] + tau * w1 /2, t[m] + tau /2)
    w3 = f(u_ERK4[m] + tau * w2 /2, t[m]+ tau /2)
    w4 = f(u_ERK4[m] + tau * w3, t[m] + tau)
    u_ERK4[m+1] = u_ERK4[m] + tau*(w1 / 6 + w2 / 3 + w3 / 3 + w4 / 6)

'''''

#graph
plt.plot(u_ERK2[:,0], u_ERK2[:,1], '-or')
plt.plot(u_ERK1[:,0], u_ERK1[:,1], '-or')
plt.xlim((0, 1.62*80)), plt.ylim(0, 80)
plt.show()
