import numpy as np
import matplotlib.pyplot as plt

x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 10     #Конец отрезка
M = 100

def f(u, x):
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f



def f_u(f,u, t, tau):
    f_u = (f(u, t + tau) - f(u, t))/tau
    return f_u

#def F_u(f, u, t, tau):
#    return np.arange([[0, 1], [2*np.sin(t)/np.cos(t), 0]])

t, tau = np.linspace(x_0, X, M + 1, retstep = True) #retstep - возвращает шаг

u_CROS1 = np.empty((M + 1, 3))

u_CROS1[0] = [y_0, v_0, w_0]

alpha= (1 + 1j)/2 # CROS1 (схема Розенброка с комплексным коэффициентом)
for m in range(M) :
    w_1= f(u_CROS1[m],t[m])/(1 - alpha*tau*f_u(f, u_CROS1[m],t[m], tau))
    u_CROS1[m + 1] = u_CROS1[m] + tau*w_1.real  



# Аналитическое решение
y_analytical = 0.5 * t**2 - np.pi * t * 0.5 + 1 + (np.pi ** 2) / 8

# Построение графика
plt.plot(t, u_CROS1[:, 0], label='Численное решение', color = 'red')
plt.plot(t, y_analytical, label='Аналитическое решение', linestyle = "-", color = 'green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('CROS1')
plt.legend()
plt.grid(True)
plt.show()