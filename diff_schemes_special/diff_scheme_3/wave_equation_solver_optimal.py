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

def cross_scheme(N=200, M=200):
    """
    Implementation of the Cross (explicit) scheme for the wave equation
    with optimal parameters.
    
    Parameters:
    N : int
        Number of nodes along x-axis
    M : int
        Number of nodes along t-axis
    """
    print("\n" + "=" * 50)
    print(f"СХЕМА 'КРЕСТ' (ЯВНАЯ СХЕМА)")
    print("=" * 50)
    
    # Grid parameters
    case_grid = 2  # Using grid with 2nd order approximation
    
    print(f"Число узлов по x (N): {N}")
    print(f"Число узлов по t (M): {M}")
    print("Используется сетка со вторым порядком аппроксимации по h")
    
    # Create grid
    h = 1/(N-1)
    x0 = -h/2
    
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
    
    # Create and save combined plot
    create_combined_plot(X, T, y, u, err, 
                       'Решение уравнения колебаний (схема "крест")', 
                       'cross_combined_optimal.png')
    
    # Calculate and print max error
    max_err = np.max(np.abs(err))
    print(f"Максимальная абсолютная погрешность: {max_err:.8f}")
    
    return y, u, err

def implicit_scheme(N=200, M=200, sigma=0.3):
    """
    Implementation of the Implicit scheme for the wave equation
    with optimal parameters.
    
    Parameters:
    N : int
        Number of nodes along x-axis
    M : int
        Number of nodes along t-axis
    sigma : float
        Weight parameter (optimal value is 1.0 based on experiments)
    """
    print("\n" + "=" * 50)
    print(f"НЕЯВНАЯ СХЕМА (σ = {sigma})")
    print("=" * 50)
    
    # Grid parameters
    case_grid = 2  # Using grid with 2nd order approximation
    
    print(f"Число узлов по x (N): {N}")
    print(f"Число узлов по t (M): {M}")
    print(f"Весовой параметр sigma: {sigma}")
    print("Используется сетка со вторым порядком аппроксимации по h")
    
    # Create grid
    h = 1/(N-1)
    x0 = -h/2
    
    x = np.zeros(N+1)
    for i in range(N+1):
        x[i] = x0 + i*h
    
    t = np.zeros(M+1)
    tau = 1/M
    for j in range(M+1):
        t[j] = j*tau
    
    # Check stability condition
    stability_condition = True  # Unconditionally stable for sigma >= 0.25
    print("Схема безусловно устойчива при σ ≥ 0.25")
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
    
    # Create and save combined plot
    create_combined_plot(X, T, y, u, err, 
                       f'Решение уравнения колебаний (неявная схема, σ={sigma})', 
                       f'implicit_combined_optimal.png')
    
    # Calculate and print max error
    max_err = np.max(np.abs(err))
    print(f"Максимальная абсолютная погрешность: {max_err:.8f}")
    
    return y, u, err

# Run simulations with optimal parameters
if __name__ == "__main__":
    print("=" * 80)
    print("ЧИСЛЕННОЕ РЕШЕНИЕ УРАВНЕНИЯ КОЛЕБАНИЙ ДЛЯ ЗАДАЧИ (Б)")
    print("∂²u/∂t² = ∂²u/∂x² на отрезке [0,1] с граничными условиями ∂u/∂x|_{x=0} = 0, ∂u/∂x|_{x=1} = 2")
    print("начальные условия: u(x,0) = x², u_t(x,0) = cos(πx)")
    print("=" * 80)
    
    # Используем оптимальные параметры на основе проведенного исследования:
    # - Сетка: N=M=200 (мелкая сетка для лучшей точности)
    # - Для схемы "крест": стандартная явная схема с sₕ = O(h²) + O(τ²)
    # - Для неявной схемы: σ=1.0 (оптимальное значение из исследования)
    
    # Запускаем расчеты с оптимальными параметрами
    cross_scheme(N=200, M=200)
    implicit_scheme(N=200, M=200, sigma=0.3)
    
    print("\nВсе расчеты завершены. Результаты сохранены в папке 'results'.") 