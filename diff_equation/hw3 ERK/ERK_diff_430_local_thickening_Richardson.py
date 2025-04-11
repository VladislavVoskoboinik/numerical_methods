#Файл в разработке
# Problem: как считать Ричардсона, если после сгущения размер сетки может измениться?

import math
import numpy as np
import matplotlib.pyplot as plt

# differential equation: y'' = 2*(y-1) * ctg(x), y(pi/2) = 1

# Параметры
x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 10     #Конец отрезка
M = 100
tau = 0.2
p = 3; q = 3; S = 3
eps = 0.1 #Дифур абстрактый, поэтому не получится прикинуть значений eps заранее

r = 2 # Коэффициент сгущения


u = [[y_0, v_0, w_0]]
u_thickened = [[y_0, v_0, w_0]]



def f(u, x):
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f

def ERK3_step(current_x, current_u, current_step):
    w_1 = f(current_u, current_x)
    w_2 = f(current_u + 1/2*current_step*w_1, current_x + 1/2 * current_step)
    w_3 = f(current_u + 3/4 * current_step * w_2, current_x + 3/4 * current_step)
    return current_u + current_step * (2/9*w_1 + 3/9 * w_2 + 4/9 * w_3)


def ERK3_local_thickening(M, eps, x_0, X):
    tau = (X - x_0) / M
    u = [[y_0, v_0, w_0]]
    u_thickened = [[y_0, v_0, w_0]]
    x = [x_0]
    m = 0
    while x[-1] < X and m <= M:
        u.append(ERK3_step(x[-1], u[-1], tau))
        u_thickened.append(ERK3_step(x[-1], u_thickened[-1], tau/r))
        u_thickened.append(ERK3_step(x[-1] + tau/r, u_thickened[-1], tau/r))
        print(u_thickened[-1], u[-1])
        #error = np.sqrt(sum((u[m][j]**2 - u_thickened[m][j] ** 2) for i in range(3) for j in range(3)))
        error = np.linalg.norm(u[-1] - u_thickened[-1])
        x.append(x[-1] + tau)
        if error > eps:  # Защита от слишком малых значений
            tau_new = ((eps * (r**p - 1) * tau**(p+1)) / (error*(X-x_0))) **(1/p)
            tau = min(tau_new, 2*tau)  # Ограничиваем максимальный рост шага
        else:
            tau = 2*tau  # Если ошибка слишком мала, увеличиваем шаг

        print(f'x[-1]={x[-1]} : tau={tau}')
        
        m += 1

    x = np.array(x)
    u = np.array(u)

    return np.array(x), np.array(u)

#x, u = ERK3_local_thickening(r * M, eps, x_0, X)

u_richardson = [[] * S] * S
R = [[] * S] * S
p_eff = [[] * S] * S
#u_richardson = np.array(data)
#u_richardson = np.zeros((S, (S, (M, 3))))

for s in range(1, S):
    x, u = ERK3_local_thickening(r**s * M, eps, x_0, X)
    u_richardson[s:0].append(u[:, 0])

print(u_richardson)
for s in range(1, S):
    for l in range(S):
        R[s, l] = (u_richardson[s, l] - u_richardson[s-1, l]) / (r ** (p + l * q) - 1)
        u_richardson[s, l+1] = u_richardson[s, l] + R[s, l]

'''''
while x[m] < X:
    w_1 = f(u[m], x[m])
    w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
    w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
    u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3) 
    
    error = np.sqrt(np.sum((u[m+1] - u_emb)**2))
    if error > eps:  # Защита от слишком малых значений
        tau_new = tau * (eps/error)**(1/(p-1))
        tau = min(tau_new, 2*tau)  # Ограничиваем максимальный рост шага
    else:
        tau = 2*tau  # Если ошибка слишком мала, увеличиваем шаг
        
    
    print(f'm={m} : tau={tau}')
    
    w_1 = f(u[m], x[m])
    w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
    w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
    u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3) 
    
    x[m+1] = x[m] + tau
    
    m = m + 1
'''''

# Аналитическое решение
x_analytical = np.linspace(x_0, X, M*4)
y_analytical = 0.5 * x_analytical**2 - np.pi * x_analytical * 0.5 + 1 + (np.pi ** 2) / 8

# Построение графика
plt.plot(x, u[:, 0], label='ERK3', marker = 'o', color = 'red')
#plt.plot(x, u_ERK2[:, 0], label = "ERK2")
#plt.plot(x, u_ERK4[:, 0], label = "ERK3")
plt.plot(x_analytical, y_analytical, label='Аналитическое решение', linestyle = "--", color = 'green')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение уравнения третьего порядка\nВыбор шага методом локального сгущения')
plt.legend()
plt.grid(True)
plt.show()
