#! python3.7
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

exp = np.exp(1)

# Определение функции, задающей начальное условие
def u_init(x):
    return 0

# Функция f подготавливает массив, содержащий элементы вектор-функции
def f(y, h, N, lamb):
    f_vec = np.zeros(3*N-3)
    f_vec[0] = -y[2*N-2] + (1/h**2)*(y[1] - 2*y[0]) - np.exp(lamb*y[0])
    f_vec[N-2] = -y[3*N-4] + (1/h**2)*(y[N-3] - 2*y[N-2]) - np.exp(lamb*y[N-2])
    
    for n in range(1, N-2):
        f_vec[n] = -y[n+2*N-2] + (1/h**2)*(y[n+1] - 2*y[n] + y[n-1]) - np.exp(lamb*y[n])
    
    for n in range(N-1, 2*N-2):
        f_vec[n] = -1 * y[n-N+1]
    
    for n in range(2*N-2, 3*N-3):
        f_vec[n] = -1 * y[n-N+1]
    
    return f_vec

# Функция подготавливает матрицу дифференциального оператора
def D(N, h):
    D_mat = np.zeros((3*N-3, 3*N-3))
    for n in range(N-1, 3*N-3):
        D_mat[n, n] = 1
    return D_mat

# Функция подготавливает матрицу Якоби
def f_y(y, h, N, lamb):
    jac = np.zeros((3*N-3, 3*N-3))
    
    for n in range(N-1, 2*N-2):
        jac[n, n-N+1] = -1
    
    for n in range(2*N-2, 3*N-3):
        jac[n, n-N+1] = 1
    
    jac[0, 1] = 1/h**2
    jac[0, 2*N-2] = -1
    jac[0, 0] = -2/h**2 - lamb*np.exp(lamb*y[0])
    
    jac[N-2, N-3] = 1/h**2
    jac[N-2, 3*N-4] = -1
    jac[N-2, N-2] = -2/h**2 - lamb*np.exp(lamb*y[N-2])
    
    for n in range(1, N-2):
        jac[n, n+1] = 1/h**2
        jac[n, n+2*N-2] = -1
        jac[n, n-1] = 1/h**2
        jac[n, n] = -2/h**2 - lamb*np.exp(lamb*y[n])
    
    return jac

# Параметры задачи
a = 0.0; b = 1.0
t_0 = 0.0; T = 30
lamb = -1
alpha = (1 + 1j)/2  # CROS1 схема
N = 20; M = 25

# Инициализация сетки
h = (b - a)/N
x = np.linspace(a, b, N+1)
tau = (T - t_0)/M
t = np.linspace(t_0, T, M+1)

# Выделение памяти
u = np.zeros((M+1, N+1))
y = np.zeros((M+1, 3*N-3))

# Начальные условия
for n in range(N+1):
    u[0, n] = u_init(x[n])

y[0, :N] = 0
y[0, N:2*N] = -lamb
y[0, 2*N:3*N-3] = -1

# Инициализация графика
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :], 'b-')
ax.set_xlim(a, b)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Real-time Solution')
plt.show(block=False)

# Основной цикл
for m in range(M):
    # Вычисление шага
    D_mat = D(N, h)
    jac = f_y(y[m], h, N, lamb)
    f_vec = f(y[m], h, N, lamb)
    
    w_1 = linalg.solve(D_mat - alpha*tau*jac, f_vec)
    y[m+1] = y[m] + tau*w_1.real
    
    # Обновление решения
    u[m+1, 0] = 0
    u[m+1, 1] = 1/4 * y[m+1, 0]
    u[m+1, 2:N+1] = y[m+1, :N-1]
    
    # Обновление графика
    line.set_ydata(u[m+1, :])
    ax.set_title(f'Time: {t[m+1]:.3f}')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()
plt.show()
