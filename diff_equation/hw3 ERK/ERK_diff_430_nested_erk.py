import math
import numpy as np
import matplotlib.pyplot as plt

# differential equation: y'' = 2*(y-1) * ctg(x), y(pi/2) = 1

# Параметры
x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 10    #Конец отрезка
M = 100
tau = 0.2
p = 3
eps = 1e-12 
#x_grid, tau =  np.linspace(x_0, X, M+1, retstep=True)
x = np.empty(M + 1)
u = np.empty((M+1, 3))
u[0] = [y_0, v_0, w_0]
x[0] = x_0
#u_ERK2 = np.empty((M+1, 3))
#u_ERK2[0] = [y_0, v_0, w_0]
#u_ERK4 = np.empty((M+1, 3))
#u_ERK4[0] = [y_0, v_0, w_0]

def f(u, x) -> np.array:
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f


#Euler
''''
for m in range(0, M):
    u[m+1] = u[m] + tau * f(u[m], x_grid[m])

    
#ERK2
for m in range(0, M):
    w_1 = f(u_ERK2[m], x_grid[m])
    w_2 = f(u_ERK2[m] + 2/3 * tau * w_1, x_grid[m] + 2/3 * tau)
    u_ERK2[m+1] = u_ERK2[m] + tau * (1/4 * w_1 + 3/4 * w_2)


#ERK4
for m in range(0, M): #ERK4
    w1 = f(u_ERK4[m], x_grid[m])
    w2 = f(u_ERK4[m] + tau * w1 /2, x_grid[m] + tau /2)
    w3 = f(u_ERK4[m] + tau * w2 /2, x_grid[m]+ tau /2)
    w4 = f(u_ERK4[m] + tau * w3, x_grid[m] + tau)
    u_ERK4[m+1] = u_ERK4[m] + tau*(w1 / 6 + w2 / 3 + w3 / 3 + w4 / 6)

'''''

m = 0
while x[m] < X and m <= M:
    w_1 = f(u[m], x[m])
    w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
    w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
    u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3) 
    
    u_emb = u[m] + tau*w_2
    
    error = np.linalg.norm(u[m + 1][0] - u_emb[0])
    print(error)
    #error = np.sqrt(np.sum((u[m+1] - u_emb)**2))
    if error > eps:  # Защита от слишком малых значений
        tau_new = tau * (eps/(error*(X-x_0)))**(1/(p-1))
        #tau = min(tau_new, 2*tau)  # Ограничиваем максимальный рост шага
    else:
        tau = 1.1*tau  # Если ошибка слишком мала, увеличиваем шаг
        
    
    print(f'm={m} : tau={tau}', x[m])
    
    w_1 = f(u[m], x[m])
    w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
    w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
    u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3) 
    
    x[m+1] = x[m] + tau
    
    m = m + 1

u = u[:m+1]
x = x[:m+1]
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
plt.title('Решение уравнения третьего порядка\nВыбор шага методом вложенной схемы')
plt.legend()
plt.grid(True)
plt.show()
