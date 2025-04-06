import numpy as np
import math 
import matplotlib.pyplot as plt

t_0 = -1
T = 2
y_00 = 3
eps = 0.02
M = 500



t, tau = np.linspace(t_0, T, M + 1, retstep = True) #retstep - возвращает шаг

u_CROS1 = np.empty(M + 1)
u_CROS2 = np.empty(M + 1)

u_CROS1[0] = y_00
u_CROS2[0] = y_00


def f(u, t):
    f = 1/eps*(t-u)*u
    return f

def f_u(u, t):
    f_u = 1/eps*(t-2*u)
    return f_u


# CROS1
alpha= (1 + 1j)/2 # CROS1 (схема Розенброка с комплексным коэффициентом)
for m in range(M) :
    w_1= f(u_CROS1[m],t[m])/(1 - alpha*tau*f_u(u_CROS1[m],t[m]))
    u_CROS1[m + 1] = u_CROS1[m] + tau*w_1.real  


a11 = 0.1j
a21 = 0.8 - 1.3j
a22 = 0.2 + 0.1j
b1 = - 0.2j
b2 = 1 - 0.9j
c21 = 1 - 0.2j #before: c21 = 0.5 - 0.2j 
# CROS2
for m in range(M) :
    w_1= f(u_CROS2[m], t[m])/(1 - a11 * tau *f(u_CROS2[m], t[m]))
    w_2= f(u_CROS2[m] + tau*(c21*w_1).real, t[m])/(1 - a22*tau*f_u(u_CROS2[m] + tau*(a21*w_1).real, t[m]))
    u_CROS2[m+1]=u_CROS2[m] + tau*(b1*w_1 + b2*w_2).real

u_analytical = np.empty(M+1)
u_analytical = eps / (1 - t)

plt.plot(t, u_analytical, '-ob', label = "Analytical")
plt.plot(t, u_CROS1, '-or', label  = "CROS1")
plt.plot(t, u_CROS2, '-og', label = "CROS2")



plt.xlim((-1, 2)), plt.ylim(0, 3)
plt.legend()
plt.show()