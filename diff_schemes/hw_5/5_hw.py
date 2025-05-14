# -*- coding: utf-8 -*-
from numpy import zeros, exp, meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
L = 1          # Длина интервала по x
T_total = 1    # Длина интервала по t
N = 100        # Число узлов по x

# Перебираем все схемы
for case_scheme in [1, 2, 3, 4]:
    # Определяем параметры сетки для схемы
    if case_scheme == 1:    # Схема а (N <= J)
        J = N + 10
    elif case_scheme == 2:  # Схема б (N >= J)
        J = 50
    elif case_scheme == 3:  # Схема в (безусловная устойчивость)
        J = N + 10
    elif case_scheme == 4:  # Схема г (безусловная устойчивость)
        J = N

    # Генерация сетки
    h = L / N
    tau = T_total / J
    
    x = zeros(N+1)
    t = zeros(J+1)
    
    for n in range(N+1):
        x[n] = -L + n*h  # x от -L до 0
    
    for j in range(J+1):
        t[j] = j*tau

    # Аналитическое решение
    u = zeros((N+1, J+1))
    for n in range(N+1):
        for j in range(J+1):
            if t[j] + x[n] <= 0:
                u[n,j] = t[j]*exp(-(x[n]+t[j]))
            else:
                u[n,j] = x[n]*(-exp(-(x[n]+t[j])) + 1) + t[j]

    # Численное решение
    y = zeros((N+1, J+1))
    y[:, 0] = 0          # Начальное условие
    y[N, :] = t[:]       # Граничное условие
    
    for j in range(J):
        if case_scheme == 1:  # Схема а
            for n in range(N, 0, -1):
                y[n-1,j+1] = y[n-1,j] + (y[n,j]-y[n-1,j])*tau/h + tau*exp(-(t[j]+x[n-1]))
                
        elif case_scheme == 2:  # Схема б
            for n in range(N, 0, -1):
                y[n-1,j+1] = -(y[n,j+1]-y[n,j])*h/tau + h*exp(-(t[j+1]+x[n])) + y[n,j+1]
                
        elif case_scheme == 3:  # Схема в
            for n in range(N-1, -1, -1):
                y[n,j+1] = (exp(-(t[j+1]+x[n]))*tau*h + y[n,j]*h + y[n+1,j+1]*tau) / (tau+h)
                
        elif case_scheme == 4:  # Схема г
            for n in range(N, 0, -1):
                term = 2*exp(-(t[j] + tau/2 + x[n] - h/2))
                y[n-1,j+1] = (term + (y[n-1,j] - y[n,j+1] + y[n,j])/tau 
                             + (y[n,j] - y[n-1,j] + y[n,j+1])/h) / (1/h + 1/tau)

    # Вычисление ошибки
    err = y - u

    # Визуализация
    T_mesh, X_mesh = meshgrid(t, x)
    
    fig = plt.figure(figsize=(21, 6))
    fig.suptitle(f'Схема {case_scheme}', fontsize=16)
    
    # Численное решение
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(T_mesh, X_mesh, y, cmap='viridis')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_title('Численное решение')

    # Аналитическое решение
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(T_mesh, X_mesh, u, cmap='plasma')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('Аналитическое решение')

    # Ошибка
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(T_mesh, X_mesh, err, cmap='coolwarm')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_title('Погрешность')

    plt.show()