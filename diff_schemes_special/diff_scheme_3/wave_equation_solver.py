import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Create directory for saving images if it doesn't exist
output_dir = "results"
try:
    os.makedirs(output_dir)
except FileExistsError:
    # Директория уже существует - это нормально
    pass

def analytical_solution(x, t):
    """
    Analytical solution for the wave equation problem (case b):
    u(x,t) = t^2 + (1/π)*sin(πt)*cos(πx) + x^2
    """
    return t**2 + (1/np.pi)*np.sin(np.pi*t)*np.cos(np.pi*x) + x**2

def create_combined_plot(X, T, numerical, analytical, error, title, filename):
    """
    Create and save a combined 3D plot with numerical solution, analytical solution, and error
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Numerical solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, T, numerical, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u')
    ax1.set_title('Численное решение')
    fig.colorbar(surf1, ax=ax1, shrink=0.6)
    
    # Analytical solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, T, analytical, cmap='plasma')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u')
    ax2.set_title('Аналитическое решение')
    fig.colorbar(surf2, ax=ax2, shrink=0.6)
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, T, error, cmap='coolwarm')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('Ошибка')
    ax3.set_title('Погрешность')
    fig.colorbar(surf3, ax=ax3, shrink=0.6)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_error_plot(grid_sizes, errors, title, filename):
    """
    Create and save a plot showing error vs grid size
    """
    plt.figure(figsize=(10, 6))
    for method, data in errors.items():
        if data:  # Check if there's data for this method
            sizes = [size for size, _ in data]
            error_vals = [err for _, err in data]
            plt.plot(sizes, error_vals, 'o-', label=method)
    
    plt.xlabel('Число узлов сетки')
    plt.ylabel('Максимальная погрешность')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def print_grid_info(h, tau, stability_condition, stability_limit=None):
    """
    Print information about the grid and stability conditions
    """
    print(f"Шаг по пространству (h): {h:.6f}")
    print(f"Шаг по времени (tau): {tau:.6f}")
    print(f"Число Куранта (tau/h): {tau/h:.6f}")
    
    if stability_limit:
        print(f"Предел устойчивости: {stability_limit:.6f}")
        
    if stability_condition:
        print("Условие устойчивости ВЫПОЛНЕНО")
    else:
        print("Условие устойчивости НЕ ВЫПОЛНЕНО")
    print("-" * 50)

def cross_scheme(N=50, M=60, case_grid=2, save_plots=True):
    """
    Implementation of the Cross (explicit) scheme for the wave equation
    
    Parameters:
    N : int
        Number of nodes along x-axis
    M : int
        Number of nodes along t-axis
    case_grid : int
        Grid type (1 for O(h) approximation, 2 for O(h^2) approximation)
    save_plots : bool
        Whether to save plots or not
    
    Returns:
    max_error : float
        Maximum absolute error of the numerical solution
    """
    print("\n" + "=" * 50)
    print(f"СХЕМА 'КРЕСТ' (ЯВНАЯ СХЕМА) - N={N}, M={M}")
    print("=" * 50)
    
    # Grid parameters
    print(f"Число узлов по x (N): {N}")
    print(f"Число узлов по t (M): {M}")
    
    # Create grid
    if case_grid == 1:  # O(h) approximation
        h = 1/N
        x0 = 0
        print("Используется сетка с первым порядком аппроксимации по h")
    else:  # O(h^2) approximation
        h = 1/(N-1)
        x0 = -h/2
        print("Используется сетка со вторым порядком аппроксимации по h")
    
    x = np.zeros(N+1)
    for i in range(N+1):
        x[i] = x0 + i*h
    
    t = np.zeros(M+1)
    tau = 1/M
    for j in range(M+1):
        t[j] = j*tau
    
    # Check stability condition: tau <= h for wave equation
    stability_condition = tau <= h
    print_grid_info(h, tau, stability_condition)
    
    # Start timing
    start_time = time.time()
    
    # Numerical solution
    y = np.zeros((M+1, N+1))
    
    # Initial conditions for case (b)
    for i in range(N+1):
        # u(x,0) = x^2
        y[0][i] = x[i]**2
        # First time layer with second-order approximation
        # Using u_t(x,0) = cos(πx) and the Taylor expansion
        y[1][i] = x[i]**2 + tau*np.cos(np.pi*x[i]) + (tau**2/2)*(-np.pi**2*np.cos(np.pi*x[i]))
    
    # Main time loop
    for j in range(1, M):
        # Interior points
        for i in range(1, N):
            y[j+1][i] = 2*y[j][i] - y[j-1][i] + (tau**2/h**2)*(y[j][i+1] - 2*y[j][i] + y[j][i-1])
        
        # Boundary conditions: du/dx|_{x=0} = 0 and du/dx|_{x=1} = 2
        y[j+1][0] = y[j+1][1]
        y[j+1][N] = y[j+1][N-1] + 2*h
    
    # End timing
    computation_time = time.time() - start_time
    print(f"Время вычислений: {computation_time:.4f} сек.")
    
    # Compute analytical solution and error
    u = np.zeros((M+1, N+1))
    err = np.zeros((M+1, N+1))
    
    for j in range(M+1):
        for i in range(N+1):
            u[j][i] = analytical_solution(x[i], t[j])
            err[j][i] = u[j][i] - y[j][i]
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(x, t)
    
    # Create and save combined plot if requested
    if save_plots:
        create_combined_plot(X, T, y, u, err, 
                           f'Решение уравнения колебаний (схема "крест", N={N}, M={M})', 
                           f'cross_combined_N{N}_M{M}.png')
    
    # Calculate and print max error
    max_err = np.max(np.abs(err))
    print(f"Максимальная абсолютная погрешность: {max_err:.8f}")
    
    return max_err

def implicit_scheme(sigma_val=0.25, N=50, M=60, case_grid=2, save_plots=True):
    """
    Implementation of the Implicit scheme for the wave equation
    
    Parameters:
    sigma_val : float
        Weight parameter for the implicit scheme (sigma >= 0.25 for unconditional stability)
    N : int
        Number of nodes along x-axis
    M : int
        Number of nodes along t-axis
    case_grid : int
        Grid type (1 for O(h) approximation, 2 for O(h^2) approximation)
    save_plots : bool
        Whether to save plots or not
        
    Returns:
    max_error : float
        Maximum absolute error of the numerical solution
    """
    print("\n" + "=" * 50)
    print(f"НЕЯВНАЯ СХЕМА (σ = {sigma_val}) - N={N}, M={M}")
    print("=" * 50)
    
    # Grid parameters
    sigma = sigma_val  # Weight parameter
    
    print(f"Число узлов по x (N): {N}")
    print(f"Число узлов по t (M): {M}")
    print(f"Весовой параметр sigma: {sigma}")
    
    # Create grid
    if case_grid == 1:  # O(h) approximation
        h = 1/N
        x0 = 0
        print("Используется сетка с первым порядком аппроксимации по h")
    else:  # O(h^2) approximation
        h = 1/(N-1)
        x0 = -h/2
        print("Используется сетка со вторым порядком аппроксимации по h")
    
    x = np.zeros(N+1)
    for i in range(N+1):
        x[i] = x0 + i*h
    
    t = np.zeros(M+1)
    tau = 1/M
    for j in range(M+1):
        t[j] = j*tau
    
    # Check stability condition
    if sigma >= 0.25:
        stability_condition = True  # Unconditionally stable
        stability_limit = None
        print("Схема безусловно устойчива при σ ≥ 0.25")
    else:
        stability_limit = h/np.sqrt(1 - 4*sigma)
        stability_condition = tau <= stability_limit
        print(f"Схема условно устойчива при σ < 0.25. Условие устойчивости: tau ≤ h/sqrt(1-4σ)")
    
    print_grid_info(h, tau, stability_condition, stability_limit)
    
    # Start timing
    start_time = time.time()
    
    # Numerical solution
    y = np.zeros((M+1, N+1))
    
    # Initial conditions for case (b)
    for i in range(N+1):
        # u(x,0) = x^2
        y[0][i] = x[i]**2
        # First time layer with second-order approximation
        # Using u_t(x,0) = cos(πx) and the Taylor expansion
        y[1][i] = x[i]**2 + tau*np.cos(np.pi*x[i]) + (tau**2/2)*(-np.pi**2*np.cos(np.pi*x[i]))
    
    # Create arrays for Thomas algorithm coefficients
    alpha = np.zeros(N)
    beta = np.zeros(N)
    
    # Set up coefficients for the tridiagonal solver
    A = sigma*tau**2/h**2
    B = A
    C = 1 + 2*A
    
    # Main time loop
    for j in range(1, M):
        # Forward sweep of Thomas algorithm
        
        # First equation using boundary condition at x=0 (y[0] = y[1])
        alpha[0] = 1.0
        beta[0] = 0.0
        
        # Interior points
        for i in range(1, N):
            # Second derivative at time j-1
            d2y_jm1 = (y[j-1][i+1] - 2*y[j-1][i] + y[j-1][i-1])/h**2
            
            # Second derivative at time j
            d2y_j = (y[j][i+1] - 2*y[j][i] + y[j][i-1])/h**2
            
            # Right-hand side calculation (free term)
            F = 2*y[j][i] - y[j-1][i] + tau**2*(sigma*d2y_jm1 + (1-2*sigma)*d2y_j)
            
            # Update recurrence coefficients
            denominator = C - A*alpha[i-1]
            alpha[i] = B/denominator
            beta[i] = (A*beta[i-1] + F)/denominator
        
        # Boundary condition at x=1: du/dx = 2
        # Express it through recurrence relation
        y[j+1][N] = (2*h + beta[N-1])/(1 - alpha[N-1])
        
        # Backward sweep of Thomas algorithm
        for i in range(N-1, -1, -1):
            y[j+1][i] = alpha[i]*y[j+1][i+1] + beta[i]
    
    # End timing
    computation_time = time.time() - start_time
    print(f"Время вычислений: {computation_time:.4f} сек.")
    
    # Compute analytical solution and error
    u = np.zeros((M+1, N+1))
    err = np.zeros((M+1, N+1))
    
    for j in range(M+1):
        for i in range(N+1):
            u[j][i] = analytical_solution(x[i], t[j])
            err[j][i] = u[j][i] - y[j][i]
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(x, t)
    
    # Create and save combined plot if requested
    if save_plots:
        create_combined_plot(X, T, y, u, err, 
                           f'Решение уравнения колебаний (неявная схема, σ={sigma}, N={N}, M={M})', 
                           f'implicit_combined_sigma_{sigma}_N{N}_M{M}.png')
    
    # Calculate and print max error
    max_err = np.max(np.abs(err))
    print(f"Максимальная абсолютная погрешность: {max_err:.8f}")
    
    return max_err

def convergence_study():
    """
    Perform convergence study by running simulations with different grid sizes
    """
    print("\n" + "=" * 60)
    print("ИССЛЕДОВАНИЕ СХОДИМОСТИ")
    print("=" * 60)
    
    # Grid sizes to test
    grid_sizes = [20, 40, 80, 160]
    
    # Dictionary to store errors for different methods
    errors = {
        'Схема "крест"': [],
        'Неявная схема (σ=0.05)': [],
        'Неявная схема (σ=0.25)': [],
        'Неявная схема (σ=0.5)': []
    }
    
    # Run cross scheme for different grid sizes
    for N in grid_sizes:
        # Ensure stability by setting M appropriately (M at least N)
        M = N
        print(f"\nТестирование сетки N={N}, M={M}")
        
        # Run cross scheme
        err = cross_scheme(N=N, M=M, save_plots=(N == grid_sizes[-1]))
        errors['Схема "крест"'].append((N, err))
        
        # Run implicit schemes with different sigma values
        for sigma in [0.05, 0.25, 0.5]:
            try:
                err = implicit_scheme(sigma_val=sigma, N=N, M=M, save_plots=(N == grid_sizes[-1]))
                errors[f'Неявная схема (σ={sigma})'].append((N, err))
            except Exception as e:
                print(f"Ошибка при выполнении неявной схемы с σ = {sigma}: {e}")
    
    # Create plot showing errors vs grid size
    create_error_plot(grid_sizes, errors, 'Сходимость методов при измельчении сетки', 'convergence_study.png')
    
    # Print summary
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ СХОДИМОСТИ")
    print("=" * 60)
    
    for method, data in errors.items():
        if data:
            print(f"\n{method}:")
            for i in range(len(data)-1):
                N1, err1 = data[i]
                N2, err2 = data[i+1]
                ratio = err1 / err2
                order = np.log(ratio) / np.log(N2/N1)
                print(f"  N: {N1} -> {N2}, Погрешность: {err1:.8f} -> {err2:.8f}, Соотношение: {ratio:.2f}, Порядок: {order:.2f}")

def optimize_sigma():
    """
    Find optimal sigma value for the implicit scheme
    """
    print("\n" + "=" * 60)
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРА SIGMA")
    print("=" * 60)
    
    N = 100
    M = 100
    sigma_values = np.linspace(0.25, 1.0, 8)  # Test sigma values between 0.25 and 1.0
    
    errors = []
    
    for sigma in sigma_values:
        err = implicit_scheme(sigma_val=sigma, N=N, M=M, save_plots=False)
        errors.append((sigma, err))
    
    # Find the optimal sigma
    optimal_sigma, min_error = min(errors, key=lambda x: x[1])
    
    # Run with optimal sigma and save the plot
    print(f"\nОптимальное значение sigma: {optimal_sigma}")
    implicit_scheme(sigma_val=optimal_sigma, N=N, M=M, save_plots=True)
    
    # Create plot showing error vs sigma
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, [err for _, err in errors], 'o-')
    plt.axvline(x=optimal_sigma, color='r', linestyle='--', label=f'Оптимальное σ = {optimal_sigma:.3f}')
    plt.xlabel('Значение параметра σ')
    plt.ylabel('Максимальная погрешность')
    plt.title('Зависимость погрешности от параметра σ')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sigma_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Run tests
if __name__ == "__main__":
    print("=" * 80)
    print("ЧИСЛЕННОЕ РЕШЕНИЕ УРАВНЕНИЯ КОЛЕБАНИЙ ДЛЯ ЗАДАЧИ (Б)")
    print("∂²u/∂t² = ∂²u/∂x² на отрезке [0,1] с граничными условиями ∂u/∂x|_{x=0} = 0, ∂u/∂x|_{x=1} = 2")
    print("начальные условия: u(x,0) = x², u_t(x,0) = cos(πx)")
    print("=" * 80)
    
    # 1. Базовые тесты с исходными параметрами
    cross_scheme()
    for sigma in [0.05, 0.25, 0.5]:
        implicit_scheme(sigma_val=sigma)
    
    # 2. Исследование сходимости при измельчении сетки
    convergence_study()
    
    # 3. Оптимизация параметра sigma
    optimize_sigma()
    
    # 4. Тест с высоким разрешением сетки для минимизации погрешности
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ С ВЫСОКИМ РАЗРЕШЕНИЕМ СЕТКИ")
    print("=" * 60)
    
    N_high = 200
    M_high = 200
    
    cross_scheme(N=N_high, M=M_high)
    implicit_scheme(sigma_val=0.5, N=N_high, M=M_high)  # Схема Кранка-Николсона (наиболее точная)
    
    print("\nВсе расчеты завершены. Результаты сохранены в папке 'results'.") 