import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def task1():
    # Вычисление коэффициентов для аппроксимации первой производной на неравномерной сетке
    h0, h1 = sp.symbols('h0 h1')
    u0, u1, u2 = sp.symbols('u0 u1 u2')
    up, upp, uppp = sp.symbols('up upp uppp')
    
    # Разложение в ряд Тейлора
    exp1 = u0 + h0*up + h0**2/2*upp + h0**3/6*uppp
    exp2 = u0 + (h0+h1)*up + (h0+h1)**2/2*upp + (h0+h1)**3/6*uppp
    
    # Ищем коэффициенты a, b, c такие что a*u0 + b*u1 + c*u2 ≈ u'(x0)
    a, b, c = sp.symbols('a b c')
    expr = a*u0 + b*exp1 + c*exp2 - up
    
    # Составляем и решаем систему уравнений, приравнивая коэффициенты при соответствующих степенях
    eq1 = sp.Eq(expr.coeff(u0), 0)
    eq2 = sp.Eq(expr.coeff(up), 0)
    eq3 = sp.Eq(a + b + c, 0)  # условие для коэффициентов
    
    solution = sp.solve([eq1, eq2, eq3], [a, b, c])
    
    # Проверка порядка аппроксимации
    result = a*u0 + b*exp1 + c*exp2
    result = result.subs(solution)
    error = sp.expand(result - up)
    
    return solution, error

# Тестирование полученной формулы на конкретной функции
def test_derivative_approx(x0, h0_values, h1_values):
    # Тестовая функция и её производная
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)
    
    solution, _ = task1()
    a_expr, b_expr, c_expr = solution[sp.symbols('a')], solution[sp.symbols('b')], solution[sp.symbols('c')]
    
    errors = np.zeros((len(h0_values), len(h1_values)))
    
    for i, h0 in enumerate(h0_values):
        for j, h1 in enumerate(h1_values):
            # Вычисляем численные значения коэффициентов
            a_val = float(a_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            b_val = float(b_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            c_val = float(c_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            
            # Вычисляем значение функции в трех точках
            u0 = f(x0)
            u1 = f(x0 + h0)
            u2 = f(x0 + h0 + h1)
            
            # Вычисляем приближенное значение производной
            df_approx = a_val * u0 + b_val * u1 + c_val * u2
            
            # Вычисляем точное значение производной
            df_exact = df(x0)
            
            # Вычисляем погрешность
            errors[i, j] = abs(df_approx - df_exact)
    
    return errors

def task2():
    # Вычисление коэффициентов для аппроксимации второй производной на неравномерной сетке
    h0, h1 = sp.symbols('h0 h1')
    u0, u1, u2 = sp.symbols('u0 u1 u2')
    up, upp, uppp = sp.symbols('up upp uppp')
    
    # Разложение в ряд Тейлора
    exp1 = u0 + h0*up + h0**2/2*upp + h0**3/6*uppp
    exp2 = u0 + (h0+h1)*up + (h0+h1)**2/2*upp + (h0+h1)**3/6*uppp
    
    # Ищем коэффициенты a, b, c такие что a*u0 + b*u1 + c*u2 ≈ u''(x0)
    a, b, c = sp.symbols('a b c')
    expr = a*u0 + b*exp1 + c*exp2 - upp
    
    # Составляем и решаем систему уравнений
    eq1 = sp.Eq(expr.coeff(u0), 0)
    eq2 = sp.Eq(expr.coeff(up), 0)
    eq3 = sp.Eq(expr.coeff(upp) - 1, 0)  # коэффициент при u'' должен быть 1
    
    solution = sp.solve([eq1, eq2, eq3], [a, b, c])
    
    # Проверка порядка аппроксимации
    result = a*u0 + b*exp1 + c*exp2
    result = result.subs(solution)
    error = sp.expand(result - upp)
    
    return solution, error

# Тестирование полученной формулы на конкретной функции
def test_second_derivative_approx(x0, h0_values, h1_values):
    # Тестовая функция и её вторая производная
    f = lambda x: np.sin(x)
    d2f = lambda x: -np.sin(x)
    
    solution, _ = task2()
    a_expr, b_expr, c_expr = solution[sp.symbols('a')], solution[sp.symbols('b')], solution[sp.symbols('c')]
    
    errors = np.zeros((len(h0_values), len(h1_values)))
    
    for i, h0 in enumerate(h0_values):
        for j, h1 in enumerate(h1_values):
            # Вычисляем численные значения коэффициентов
            a_val = float(a_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            b_val = float(b_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            c_val = float(c_expr.subs({sp.symbols('h0'): h0, sp.symbols('h1'): h1}))
            
            # Вычисляем значение функции в трех точках
            u0 = f(x0)
            u1 = f(x0 + h0)
            u2 = f(x0 + h0 + h1)
            
            # Вычисляем приближенное значение второй производной
            d2f_approx = a_val * u0 + b_val * u1 + c_val * u2
            
            # Вычисляем точное значение второй производной
            d2f_exact = d2f(x0)
            
            # Вычисляем погрешность
            errors[i, j] = abs(d2f_approx - d2f_exact)
    
    return errors

def task3():
    # Применение формул для численного дифференцирования и оценка погрешности
    x0 = np.pi/4  # Точка, в которой вычисляем производные
    h0_values = [0.1, 0.05, 0.025, 0.0125]
    h1_values = [0.1, 0.05, 0.025, 0.0125]
    
    # Вычисление погрешностей
    errors_df = test_derivative_approx(x0, h0_values, h1_values)
    errors_d2f = test_second_derivative_approx(x0, h0_values, h1_values)
    
    # Построение графиков
    plt.figure(figsize=(12, 10))
    
    # График погрешности первой производной
    plt.subplot(2, 1, 1)
    for j, h1 in enumerate(h1_values):
        plt.loglog(h0_values, errors_df[:, j], 'o-', label=f'h1 = {h1}')
    
    # Для сравнения добавим линию O(h^2)
    ref_line = np.array([h**2 for h in h0_values]) * errors_df[0, 0] / h0_values[0]**2
    plt.loglog(h0_values, ref_line, 'k--', label='O(h²)')
    
    plt.xlabel('h0')
    plt.ylabel('Погрешность')
    plt.title('Погрешность аппроксимации первой производной')
    plt.grid(True)
    plt.legend()
    
    # График погрешности второй производной
    plt.subplot(2, 1, 2)
    for j, h1 in enumerate(h1_values):
        plt.loglog(h0_values, errors_d2f[:, j], 'o-', label=f'h1 = {h1}')
    
    # Для сравнения добавим линию O(h)
    ref_line = np.array([h for h in h0_values]) * errors_d2f[0, 0] / h0_values[0]
    plt.loglog(h0_values, ref_line, 'k--', label='O(h)')
    
    plt.xlabel('h0')
    plt.ylabel('Погрешность')
    plt.title('Погрешность аппроксимации второй производной')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('task3_errors.png')
    plt.show()
    
    return errors_df, errors_d2f

def task4a():
    """
    Решение краевой задачи:
    -u''(x) + p(x)*u'(x) + q(x)*u(x) = f(x), x ∈ [0, 1]
    u(0) = u(1) = 0
    """
    # Задаем параметры задачи
    a, b = 0, 1
    
    # Коэффициенты уравнения (пример: -u'' + u = x)
    p = lambda x: 0
    q = lambda x: 1
    f = lambda x: x
    
    # Граничные условия
    alpha, beta = 0, 0
    
    # Аналитическое решение
    u_exact = lambda x: x - np.sinh(x) / np.sinh(1)
    
    # Решение задачи для разных размеров сетки
    n_values = [10, 20, 40, 80, 160]
    errors = []
    
    for n in n_values:
        # Создаем равномерную сетку
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        
        # Формируем СЛАУ
        A = np.zeros((n+1, n+1))
        b_vec = np.zeros(n+1)
        
        # Внутренние узлы: -u''(x) + p(x)*u'(x) + q(x)*u(x) = f(x)
        for i in range(1, n):
            A[i, i-1] = 1/h**2 - p(x[i])/(2*h)  # коэффициент при u_{i-1}
            A[i, i] = -2/h**2 + q(x[i])         # коэффициент при u_i
            A[i, i+1] = 1/h**2 + p(x[i])/(2*h)  # коэффициент при u_{i+1}
            b_vec[i] = f(x[i])
        
        # Граничные условия
        A[0, 0] = 1
        b_vec[0] = alpha
        A[n, n] = 1
        b_vec[n] = beta
        
        # Решаем СЛАУ
        u_numerical = np.linalg.solve(A, b_vec)
        
        # Вычисляем аналитическое решение
        u_analytical = u_exact(x)
        
        # Вычисляем погрешность
        error = np.max(np.abs(u_numerical - u_analytical))
        errors.append(error)
        
        # Для последней итерации строим графики
        if n == n_values[-1]:
            plt.figure(figsize=(12, 10))
            
            # График численного и аналитического решений
            plt.subplot(2, 1, 1)
            plt.plot(x, u_analytical, 'b-', label='Аналитическое решение')
            plt.plot(x, u_numerical, 'r--', label='Численное решение')
            plt.grid(True)
            plt.legend()
            plt.title(f'Решение краевой задачи (n = {n})')
            
            # График погрешности
            plt.subplot(2, 1, 2)
            plt.semilogy(x, np.abs(u_numerical - u_analytical), 'g-')
            plt.grid(True)
            plt.title(f'Погрешность численного решения (max = {error:.2e})')
            plt.xlabel('x')
            plt.ylabel('|u_численное - u_аналитическое|')
            
            plt.tight_layout()
            plt.savefig('task4a_solution.png')
            plt.show()
    
    # График сходимости
    plt.figure(figsize=(8, 6))
    
    # Вычисляем порядок сходимости
    h_values = [1/n for n in n_values]
    log_h = np.log10(h_values)
    log_err = np.log10(errors)
    slope, intercept = np.polyfit(log_h, log_err, 1)
    
    plt.loglog(h_values, errors, 'bo-', label='Погрешность')
    plt.loglog(h_values, [c * h**slope for h in h_values], 'k--', 
              label=f'O(h^{slope:.2f})')
    
    plt.grid(True)
    plt.xlabel('Шаг сетки h')
    plt.ylabel('Максимальная погрешность')
    plt.title(f'Исследование сходимости (порядок ~ {slope:.2f})')
    plt.legend()
    
    plt.savefig('task4a_convergence.png')
    plt.show()
    
    return errors, slope

# Запуск всех заданий
if __name__ == "__main__":
    solution1, error1 = task1()
    print("Задание 1: Коэффициенты для аппроксимации первой производной:")
    sol1 = solution1[0]
    print(f"a = {sol1[sp.symbols('a')]}")

    print(f"b = {solution1[sp.symbols('b')]}")
    print(f"c = {solution1[sp.symbols('c')]}")
    print(f"Погрешность аппроксимации: {error1}")
    
    solution2, error2 = task2()
    print("\nЗадание 2: Коэффициенты для аппроксимации второй производной:")
    print(f"a = {solution2[sp.symbols('a')]}")
    print(f"b = {solution2[sp.symbols('b')]}")
    print(f"c = {solution2[sp.symbols('c')]}")
    print(f"Погрешность аппроксимации: {error2}")
    
    errors_df, errors_d2f = task3()
    
    errors_4a, slope_4a = task4a()
    print("\nЗадание 4а: Результаты решения краевой задачи")
    print(f"Порядок сходимости: {slope_4a:.2f}")
    for n, err in zip([10, 20, 40, 80, 160], errors_4a):
        print(f"n = {n}, погрешность = {err:.2e}")
