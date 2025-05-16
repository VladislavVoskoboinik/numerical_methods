import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

L = 1
T = 1
N = 100

def analytical_solution(x, t):
    u = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] + t <= 0:
            u[i] = t * np.exp(-(x[i] + t))
        else:
            u[i] = x[i] * (-np.exp(-(x[i] + t)) + 1) + t
    return u

def scheme_a(x, t):
    N = len(x) - 1
    J = len(t) - 1
    h = abs(x[1] - x[0])
    tau = t[1] - t[0]
    
    if tau > h:
        print("Схема А: Нарушено условие устойчивости, tau <= h")
    
    y = np.zeros((N+1, J+1))
    
    for n in range(N+1):
        y[n, 0] = 0
    
    for j in range(J+1):
        y[N, j] = t[j]
    
    for j in range(J):
        for n in range(N, 0, -1):
            y[n-1, j+1] = y[n-1, j] + (y[n, j] - y[n-1, j]) * tau / h + tau * np.exp(-(x[n-1] + t[j]))
    
    return y

def scheme_b(x, t):
    N = len(x) - 1
    J = len(t) - 1
    h = abs(x[1] - x[0])
    tau = t[1] - t[0]
    
    
    y = np.zeros((N+1, J+1))
    
    for n in range(N+1):
        y[n, 0] = 0
    
    for j in range(J+1):
        y[N, j] = t[j]
    
    for j in range(J):
        for n in range(N, 0, -1):
            y[n-1, j+1] = -(y[n, j+1] - y[n, j]) * h / tau + h * np.exp(-(t[j+1] + x[n])) + y[n, j+1]
    
    return y

def scheme_c(x, t):
    N = len(x) - 1
    J = len(t) - 1
    h = abs(x[1] - x[0])
    tau = t[1] - t[0]
    
    if tau < h:
        print("Схема В: Нарушено условие устойчивости, tau >= h")
    
    y = np.zeros((N+1, J+1))
    
    for n in range(N+1):
        y[n, 0] = 0
    
    for j in range(J+1):
        y[N, j] = t[j]
    
    for j in range(J):
        for n in range(N-1, -1, -1):
            y[n, j+1] = (np.exp(-(x[n] + t[j+1])) * tau * h + y[n, j] * h + y[n+1, j+1] * tau) / (tau + h)
    
    return y

def scheme_d(x, t):
    N = len(x) - 1
    J = len(t) - 1
    h = abs(x[1] - x[0])
    tau = t[1] - t[0]
    
    
    y = np.zeros((N+1, J+1))
    
    for n in range(N+1):
        y[n, 0] = 0
    
    for j in range(J+1):
        y[N, j] = t[j]
    
    # Схема Кранка-Николсона (реализация без использования формулы из примера)
    # Используем симметричные разностные производные по пространству и времени
    for j in range(J):
        for n in range(N, 0, -1):
            # Вычисляем значение функции-источника в центре шаблона
            center_x = (x[n] + x[n-1]) / 2
            center_t = (t[j] + t[j+1]) / 2
            f_center = np.exp(-(center_x + center_t))
            
            # Разностная аппроксимация производной по времени
            dt_term = (y[n-1, j+1] - y[n-1, j]) / tau
            
            # Усреднённая разностная аппроксимация производной по x на двух временных слоях
            dx_term_j = (y[n, j] - y[n-1, j]) / h
            dx_term_j1 = (y[n, j+1] - y[n-1, j+1]) / h
            dx_term = 0.5 * (dx_term_j + dx_term_j1)
            
            # Кранка-Николсона: dt_term - (-dx_term) = f_center
            # Выражаем y[n-1, j+1]
            y[n-1, j+1] = y[n-1, j] + 0.5 * tau * (y[n, j+1] - y[n-1, j+1] + y[n, j] - y[n-1, j]) / h + tau * f_center
            
            # Приводим к виду, где y[n-1, j+1] выражено явно
            y[n-1, j+1] = (y[n-1, j] + 0.5 * tau * (y[n, j+1] + y[n, j] - y[n-1, j]) / h + tau * f_center) / (1 + 0.5 * tau / h)
    
    return y

def solve_and_plot(scheme_num):
    plt.figure(figsize=(15, 12))
    
    J = 0
    if scheme_num == 1:
        J = 120
        scheme_name = "Схема а"
        scheme_func = scheme_a
    elif scheme_num == 2:
        J = 50
        scheme_name = "Схема б"
        scheme_func = scheme_b
    elif scheme_num == 3:
        J = 80
        scheme_name = "Схема в"
        scheme_func = scheme_c
    elif scheme_num == 4:
        J = 500
        scheme_name = "Схема г"
        scheme_func = scheme_d
    
    tau = 1.0/J
    t = np.linspace(0, 1.0, J+1)
    
    x = np.linspace(-1, 0, N+1)
    
    u_analytical = np.zeros((N+1, J+1))
    for j in range(J+1):
        u_analytical[:, j] = analytical_solution(x, t[j])
    
    u_numerical = scheme_func(x, t)
    
    error = u_numerical - u_analytical
    
    X, Tm = np.meshgrid(x, t)
    X = X.T
    Tm = Tm.T
    
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(Tm, X, u_analytical, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('u')
    ax1.set_title(f'Аналитическое решение')
    
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(Tm, X, u_numerical, cmap=cm.plasma, alpha=0.7)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u')
    ax2.set_title(f'Численное решение - {scheme_name}')
    
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(Tm, X, error, cmap=cm.coolwarm, alpha=0.7)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_zlabel('error')
    ax3.set_title(f'Погрешность 3D - {scheme_name}')
    
    plt.tight_layout()
    plt.savefig(f'scheme_{scheme_num}_solution.png')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(Tm, X, error, cmap=cm.coolwarm, alpha=0.9, 
                        linewidth=0, antialiased=True)
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('error')
    ax.set_title(f'Погрешность в 3D - {scheme_name}')
    plt.savefig(f'scheme_{scheme_num}_error_3d.png')
    plt.show()
    
    max_error = np.max(np.abs(error))
    print(f"{scheme_name}: Максимальная погрешность = {max_error:.6e}")
    
    return u_numerical, u_analytical, error



scheme_to_show = int(input("Введите номер схемы для отображения (1-4) или 0 для всех схем: "))

if scheme_to_show == 0:
    for scheme_num in range(1, 5):
        print(f"\nРешение схемой #{scheme_num}")
        u_num, u_an, err = solve_and_plot(scheme_num)
else:
    if 1 <= scheme_to_show <= 4:
        print(f"\nРешение схемой #{scheme_to_show}")
        u_num, u_an, err = solve_and_plot(scheme_to_show)
    else:
        print("Неверный номер схемы. Доступны схемы 1-4.")
