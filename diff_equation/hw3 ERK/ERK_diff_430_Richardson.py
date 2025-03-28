#in development

import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# differential equation: y'' = 2*(y-1) * ctg(x), y(pi/2) = 1

# Параметры
x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 10     #Конец отрезка
M = 50
tau = 0.2
p = 3; q = 3; S = 3
eps = 1e-12 #Дифур абстрактый, поэтому не получится прикинуть значений eps заранее

r = 2 # Коэффициент сгущения


u = [[y_0, v_0, w_0]]
u_0 = [y_0, v_0, w_0]


def f(u, x) -> np.array: 
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f

def ERK3_step(current_x, current_u, current_step) -> np.array:
    w_1 = f(current_u, current_x)
    w_2 = f(current_u + 1/2*current_step*w_1, current_x + 1/2 * current_step)
    w_3 = f(current_u + 3/4 * current_step * w_2, current_x + 3/4 * current_step)
    return current_u + current_step * (2/9*w_1 + 3/9 * w_2 + 4/9 * w_3)


def ERK3(M, u_0, x) -> tuple[np.array, np.array]:
    #x, tau = np.linspace(x_0, X, M+1, retstep = True) 
    u = np.empty((M+1, 3))
    tau = (x[1]-x[0]) / M
    u[0] = u_0
    for m in range(0, len(x)): #ERK3
        w1 = f(u[m], x[m])
        w2 = f(u[m] + tau * w1 /2, x[m] + tau /2)
        w3 = f(u[m] + tau * w2 /2, x[m]+ tau /2)
        w4 = f(u[m] + tau * w3, x[m] + tau)
        u[m+1] = u[m] + tau*(w1 / 6 + w2 / 3 + w3 / 3 + w4 / 6)
    
    return x, u


def ERK3_local_thickening(M, eps, x_0, X) -> tuple[np.array, np.array]:
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
        error = np.linalg.norm(u[-1][0] - u_thickened[-1][0])

        x.append(x[-1] + tau)
        if error > eps:  # Защита от слишком малых значений
            tau = ((eps * (r**p - 1) * tau**(p+1)) / (error*(X-x_0))) **(1/p)
            #tau = min(tau_new, 1.1*tau) 
        else:
            tau = 1.1*tau  # Если ошибка слишком мала, увеличиваем шаг

        print(f'x[-1]={x[-1]} : tau={tau}')
        
        m += 1

    x = np.array(x)
    u = np.array(u)

    return np.array(x), np.array(u)


def richardson_extrapolation_ERK3_local_thickening(f, u_0, x_0, X, epsilon=1e-6, r=2, p=3, q=1, S=5, M_0=2) -> list[np.array, np.array, dict] :
    """
    Реализация метода сгущения сеток Ричардсона для ERK3
    dict
    Параметры:
    f - функция правой части ОДУ
    u_0 - начальные условия [y0, v0, w0]
    x_0 - начальная точка
    X - конечная точка
    epsilon - требуемая точность
    r - коэффициент сгущения сетки
    p - порядок метода
    q - параметр увеличения порядка
    S - максимальное число уровней
    N0 - начальное число шагов
    
    Возвращает:
    x - массив узлов последней сетки
    u - уточненное решение
    tables - данные для таблиц
    """
    # Инициализация треугольных массивов
    U = np.zeros((S, S, 3), dtype=np.float64)
    R = np.zeros((S, S, 3), dtype=np.float64)
    p_eff = np.zeros((S, S), dtype=np.float64)
    
    solutions = []
    converged = False
    x, u = ERK3_local_thickening(M_0, eps, x_0, X)
    solutions.append((x, u))
    U[0, 0] = u[-1]
    M_0 = len(x)
    for s in range(1, S):
        M = r**s * M_0
        x, u = ERK3(M, u_0, x)
        solutions.append((x, u[:r**s]))
        U[s,0] = u[-1]  # Сохраняем только конечную точку
        
        # Проверка сходимости
        if s > 0:
            error = np.max(np.abs(U[s,0] - U[s-1,0]))
            if error < epsilon:
                converged = True
                S = s+1
                break
        
        # Экстраполяция Ричардсона
        for l in range(s):
            denominator = r**(p + l*q) - 1
            R[s,l] = (U[s,l] - U[s-1,l]) / denominator
            U[s,l+1] = U[s,l] + R[s,l]

    # Расчет эффективных порядков
    for s in range(3, S):
        for l in range(s-1):
            ratio = np.abs(R[s-1,l] / R[s,l])
            p_eff[s,l] = np.log(ratio) / np.log(r) if R[s,l].any() != 0 else 0

    # Форматирование таблиц
    def prepare_table(data, val_format):
        table = []
        for s in range(S):
            row = [f"s={s}"]
            for l in range(s+1):
                if l < data.shape[1]:
                    row.append(val_format(data[s,l]))
            table.append(row)
        return table

    tables = {
        "solutions": prepare_table(U[:,:,0], lambda x: f"{x:.16f}"),
        "errors": prepare_table(R[:,:,0], lambda x: f"{x:.2e}"),
        "orders": prepare_table(p_eff, lambda x: f"{x:.2f}")
    }

    # Построение графиков
    plt.figure(figsize=(12, 6))
    for s in range(S):
        x, u = solutions[s]
        plt.plot(x, u[:,0], label=f'N={r**s*M_0}', alpha=0.7)
    plt.title('Решение методом сгущения сеток')
    # Аналитическое решение
    y_analytical = 0.5 * x**2 - np.pi * x * 0.5 + 1 + (np.pi ** 2) / 8
    plt.plot(x, y_analytical, label='Аналитическое решение', linestyle = "--", color = 'green')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.grid(True)
    
    return x, u, tables

x, u, tables = richardson_extrapolation_ERK3_local_thickening(f, u_0, x_0, X, eps, r, p, q, S, M)

x_analytical = np.linspace(x_0, X, M*16)
y_analytical = 0.5 * x_analytical**2 - np.pi * x_analytical * 0.5 + 1 + (np.pi ** 2) / 8


print("\nТаблица приближенных значений (y-компонента):")
print(tabulate(tables["solutions"], headers=["Level"]+[f"l={l}" for l in range(S)], tablefmt="grid"))

print("\nТаблица оценок ошибок:")
print(tabulate(tables["errors"], headers=["Level"]+[f"l={l}" for l in range(S)], tablefmt="grid"))

print("\nТаблица эффективных порядков:")
print(tabulate(tables["orders"], headers=["Level"]+[f"l={l}" for l in range(S)], tablefmt="grid"))

plt.show()