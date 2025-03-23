import numpy as np
import matplotlib.pyplot as plt

def f(u, x):
    """Правая часть системы дифференциальных уравнений"""
    f = np.empty(3)
    f[0] = u[1]  # y' = v
    f[1] = u[2]  # v' = w
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x)  # w' = 2*(w-1)*ctg(x)
    return f

def solve_erk3(x_0, y_0, v_0, w_0, X, eps, p=3):
    """
    Решение системы с помощью ERK3 с автоматическим выбором шага
    
    Параметры:
    x_0: начальное значение x
    y_0, v_0, w_0: начальные значения y, y', y''
    X: конечное значение x
    eps: требуемая точность
    p: порядок метода
    """
    # Инициализация
    M = 10000  # Максимальное количество шагов
    tau_initial = 0.01  # Начальный шаг
    tau = tau_initial
    
    # Инициализация массивов
    x = np.zeros(M + 1)
    u = np.zeros((M + 1, 3))
    x[0] = x_0
    u[0] = [y_0, v_0, w_0]
    
    m = 0
    while x[m] < X and m < M:
        # Основной расчет ERK3
        w_1 = f(u[m], x[m])
        w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
        w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
        u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3)
        
        # Расчет вложенного решения для оценки ошибки
        u_emb = u[m] + tau*w_2
        
        # Расчет ошибки и адаптивного шага
        error = np.sqrt(np.sum((u[m+1] - u_emb)**2))
        if error > eps:  # Защита от слишком малых значений
            tau_new = tau * (eps/error)**(1/(p-1))
            tau = min(tau_new, 2*tau)  # Ограничиваем максимальный рост шага
        else:
            tau = 2*tau  # Если ошибка слишком мала, увеличиваем шаг
        w_1 = f(u[m], x[m])
        w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2*tau)
        w_3 = f(u[m] + 3/4*tau*w_2, x[m] + 3/4*tau)
        u[m+1] = u[m] + tau*(2/9*w_1 + 3/9*w_2 + 4/9*w_3)    
        x[m+1] = x[m] + tau
        m += 1
        
        # Вывод прогресса каждые 100 шагов
        if m % 100 == 0:
            print(f'Шаг {m}: x = {x[m]:.3f}, tau = {tau:.6f}, error = {error:.6f}')
    
    # Обрезаем массивы до фактически использованной длины
    x = x[:m+1]
    u = u[:m+1]
    return x, u

def main():
    # Параметры задачи
    x_0 = np.pi/2  # Начальное значение x
    y_0 = 1        # Начальное значение y
    v_0 = 0        # Начальное значение y'
    w_0 = 1        # Начальное значение y''
    X = 10   # Конец отрезка
    eps = 0.1      # Требуемая точность
    
    # Решение системы
    x, u = solve_erk3(x_0, y_0, v_0, w_0, X, eps)
    
    # Аналитическое решение
    y_analytical = 0.5 * x**2 - np.pi * x * 0.5 + 1 + (np.pi ** 2) / 8
    
    # Построение графиков
    plt.figure(figsize=(12, 8))
    
    # График решения
    plt.subplot(2, 1, 1)
    plt.plot(x, u[:, 0], 'r-', label='ERK3')
    plt.plot(x, y_analytical, 'g--', label='Аналитическое решение')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение уравнения третьего порядка')
    plt.legend()
    plt.grid(True)
    
    # График ошибки
    plt.subplot(2, 1, 2)
    plt.plot(x, np.abs(u[:, 0] - y_analytical), 'b-', label='Абсолютная ошибка')
    plt.xlabel('x')
    plt.ylabel('|y - y_analytical|')
    plt.title('Ошибка численного решения')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print(f'\nСтатистика решения:')
    print(f'Количество шагов: {len(x)-1}')
    print(f'Максимальная ошибка: {np.max(np.abs(u[:, 0] - y_analytical)):.6f}')
    print(f'Средняя ошибка: {np.mean(np.abs(u[:, 0] - y_analytical)):.6f}')

if __name__ == "__main__":
    main()
