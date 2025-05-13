import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def analytical_solution(x, y, t):
    """
    Analytical solution for the self-study problem.
    
    The problem:
    ∂²u/∂t² = Δu + 2(x + y), x ∈ (0, 1), y ∈ (0, 1), t ∈ (0, 1]
    ∂u/∂x|_{x=0} = t², u|_{x=1} = t²(1 + y)
    ∂u/∂y|_{y=0} = t², u|_{y=1} = t²(x + 1)
    u|_{t=0} = 5 cos(πx/2) cos(3πy/2)
    ∂u/∂t|_{t=0} = 0
    """
    # Аналитическое решение для уравнения волны с источником и заданными граничными условиями
    lambda_mn = (np.pi/2)**2 + (3*np.pi/2)**2  # = 5π²/2
    omega = np.sqrt(lambda_mn)
    
    # Полное решение состоит из однородной части и частного решения для источника
    return 5 * np.cos(omega * t) * np.cos(np.pi * x / 2) * np.cos(3 * np.pi * y / 2) + t**2 * (x + y)


def cross_scheme_2d(N, M, T, a=1.0):
    """
    Implementation of the Cross scheme for 2D wave equation.
    
    Parameters:
    -----------
    N : int
        Number of grid points in x direction
    M : int
        Number of grid points in y direction
    T : float
        Final time
    a : float
        Wave speed coefficient (default=1.0)
    
    Returns:
    --------
    w : ndarray
        Solution array at the final time step
    x_grid : ndarray
        Grid points in x direction
    y_grid : ndarray
        Grid points in y direction
    dt : float
        Time step
    J : int
        Number of time steps
    all_solutions : list
        List of solutions at all time steps
    """
    # Grid spacing (with offset)
    hx = 1.0 / (N - 0.5)
    hy = 1.0 / (M - 0.5)
    
    # Create grid points with offset
    x = np.zeros(N+1)
    y = np.zeros(M+1)
    
    for n in range(N+1):
        x[n] = -hx/2 + n*hx
    
    for m in range(M+1):
        y[m] = -hy/2 + m*hy
    
    # Compute time step for stability
    dt_stability = 1.0 / (a * np.sqrt(1/hx**2 + 1/hy**2))
    J = 2 + int(np.round(T / dt_stability))  # Ensure enough time steps
    dt = T / J
    
    # Создаем массив временных точек
    t = np.zeros(J+1)
    for j in range(J+1):
        t[j] = j * dt
    
    print(f"Cross scheme: N={N}, M={M}, J={J}")
    print(f"hx={hx:.6f}, hy={hy:.6f}, dt={dt:.6f}")
    
    # Initialize solution array
    w = np.zeros((M+1, N+1, J+1))
    err = np.zeros((M+1, N+1, J+1))  # массив для ошибки
    
    # Set initial condition u(x,y,0)
    for n in range(N+1):
        for m in range(M+1):
            # Используем только однородную часть, т.к. частное решение равно 0 при t=0
            w[m, n, 0] = 5 * np.cos(np.pi * x[n] / 2) * np.cos(3 * np.pi * y[m] / 2)
            # Вычисляем ошибку для начального условия
            err[m, n, 0] = w[m, n, 0] - analytical_solution(x[n], y[m], 0)
    
    # Set initial derivative du/dt(x,y,0) = 0 with second-order accuracy
    # Используем уравнение волны: ∂²u/∂t² = Δu + 2(x + y) при t=0
    lambda_mn = (np.pi/2)**2 + (3*np.pi/2)**2
    omega = np.sqrt(lambda_mn)
    
    for n in range(N+1):
        for m in range(M+1):
            # Для t=dt используем разложение в ряд Тейлора с учетом ∂u/∂t|_{t=0} = 0
            # u(x,y,dt) ≈ u(x,y,0) + dt * (∂u/∂t)|_{t=0} + (dt²/2) * (∂²u/∂t²)|_{t=0}
            # (∂²u/∂t²)|_{t=0} = Δu(x,y,0) + 2(x + y)
            
            # Вычисляем Лапласиан в точке (x,y,0)
            laplacian_u0 = 0
            if n > 0 and n < N and m > 0 and m < M:
                laplacian_u0 = ((w[m, n+1, 0] - 2*w[m, n, 0] + w[m, n-1, 0]) / hx**2 + 
                              (w[m+1, n, 0] - 2*w[m, n, 0] + w[m-1, n, 0]) / hy**2)
            else:
                # Для граничных точек используем аналитическое выражение для Лапласиана
                laplacian_u0 = -5 * omega**2 * np.cos(np.pi * x[n] / 2) * np.cos(3 * np.pi * y[m] / 2)
            
            # Вычисляем значение на первом временном слое
            w[m, n, 1] = w[m, n, 0] + 0.5 * dt**2 * (laplacian_u0 + 2*(x[n] + y[m]))
            
            # Вычисляем ошибку для первого шага по времени
            err[m, n, 1] = w[m, n, 1] - analytical_solution(x[n], y[m], t[1])
    
    # Time stepping using Cross scheme
    all_solutions = [w[:, :, 0].copy(), w[:, :, 1].copy()]
    all_errors = [err[:, :, 0].copy(), err[:, :, 1].copy()]
    
    for j in range(1, J):
        t_next = t[j+1]
        
        # Interior points
        for n in range(1, N):
            for m in range(1, M):
                w_xx = (w[m, n+1, j] - 2*w[m, n, j] + w[m, n-1, j]) / hx**2
                w_yy = (w[m+1, n, j] - 2*w[m, n, j] + w[m-1, n, j]) / hy**2
                w[m, n, j+1] = 2*w[m, n, j] - w[m, n, j-1] + dt**2 * (w_xx + w_yy + 2*(x[n] + y[m]))
                # Вычисляем ошибку для внутренних точек
                err[m, n, j+1] = w[m, n, j+1] - analytical_solution(x[n], y[m], t_next)
        
        # Apply boundary conditions
        
        # Bottom boundary (y=0): Neumann condition ∂u/∂y = t²
        for n in range(N+1):
            if n == 0 or n == N:
                continue  # Углы обрабатываем отдельно
            # Используем 3-точечную аппроксимацию для производной
            # (u_{i,1} - u_{i,-1})/(2*hy) = t²
            # u_{i,-1} - фиктивный узел, выражаем: u_{i,-1} = u_{i,1} - 2*hy*t²
            # Тогда u_{i,0} = (u_{i,1} + u_{i,-1})/2 = u_{i,1} - hy*t²
            w[0, n, j+1] = w[1, n, j+1] - hy * t_next**2
            err[0, n, j+1] = w[0, n, j+1] - analytical_solution(x[n], y[0], t_next)
        
        # Top boundary (y=1): Dirichlet condition u|_{y=1} = t²(x + 1)
        for n in range(N+1):
            if n == 0 or n == N:
                continue  # Углы обрабатываем отдельно
            w[M, n, j+1] = t_next**2 * (x[n] + 1)
            err[M, n, j+1] = w[M, n, j+1] - analytical_solution(x[n], y[M], t_next)
        
        # Left boundary (x=0): Neumann condition ∂u/∂x = t²
        for m in range(M+1):
            if m == 0 or m == M:
                continue  # Углы обрабатываем отдельно
            w[m, 0, j+1] = w[m, 1, j+1] - hx * t_next**2
            err[m, 0, j+1] = w[m, 0, j+1] - analytical_solution(x[0], y[m], t_next)
        
        # Right boundary (x=1): Dirichlet condition u|_{x=1} = t²(1 + y)
        for m in range(M+1):
            if m == 0 or m == M:
                continue  # Углы обрабатываем отдельно
            w[m, N, j+1] = t_next**2 * (1 + y[m])
            err[m, N, j+1] = w[m, N, j+1] - analytical_solution(x[N], y[m], t_next)
        
        # Corner points - корректная обработка углов
        # Bottom-left corner (x=0, y=0): ∂u/∂x = t², ∂u/∂y = t²
        # Здесь мы должны усреднить решения с учетом обоих граничных условий
        w[0, 0, j+1] = (w[0, 1, j+1] - hx * t_next**2 + w[1, 0, j+1] - hy * t_next**2) / 2
        
        # Bottom-right corner (x=1, y=0): u = t²(1 + y), ∂u/∂y = t²
        # Условие Дирихле имеет приоритет
        w[0, N, j+1] = t_next**2 * (1 + y[0])
        
        # Top-left corner (x=0, y=1): ∂u/∂x = t², u = t²(x + 1)
        # Условие Дирихле имеет приоритет
        w[M, 0, j+1] = t_next**2 * (x[0] + 1)
        
        # Top-right corner (x=1, y=1): u = t²(1 + y) = t²(1 + 1) = 2t²
        w[M, N, j+1] = t_next**2 * (1 + y[M])
        
        # Ошибки в угловых точках
        err[0, 0, j+1] = w[0, 0, j+1] - analytical_solution(x[0], y[0], t_next)
        err[0, N, j+1] = w[0, N, j+1] - analytical_solution(x[N], y[0], t_next)
        err[M, 0, j+1] = w[M, 0, j+1] - analytical_solution(x[0], y[M], t_next)
        err[M, N, j+1] = w[M, N, j+1] - analytical_solution(x[N], y[M], t_next)
        
        all_solutions.append(w[:, :, j+1].copy())
        all_errors.append(err[:, :, j+1].copy())
        
        # Выводим максимальную ошибку на каждом временном слое для мониторинга
        if j % 20 == 0 or j == J-1:
            print(f"Time step {j}, max error: {np.max(np.abs(err[:, :, j+1])):.6e}")
    
    return w[:, :, J], x, y, dt, J, all_solutions, err[:, :, J]


def factorized_scheme_2d(N, M, T, sigma=0.25, a=1.0):
    """
    Implementation of the Evolutionary Factorization scheme for 2D wave equation.
    
    Parameters:
    -----------
    N : int
        Number of grid points in x direction
    M : int
        Number of grid points in y direction
    T : float
        Final time
    sigma : float
        Weight parameter (σ ≥ 0.25 for unconditional stability)
    a : float
        Wave speed coefficient (default=1.0)
    
    Returns:
    --------
    w : ndarray
        Solution array at the final time step
    x_grid : ndarray
        Grid points in x direction
    y_grid : ndarray
        Grid points in y direction
    dt : float
        Time step
    J : int
        Number of time steps
    all_solutions : list
        List of solutions at all time steps
    """
    # Grid spacing (with offset)
    hx = 1.0 / (N - 0.5)
    hy = 1.0 / (M - 0.5)
    
    # Create grid points with offset
    x = np.zeros(N+1)
    y = np.zeros(M+1)
    
    for n in range(N+1):
        x[n] = -hx/2 + n*hx
    
    for m in range(M+1):
        y[m] = -hy/2 + m*hy
    
    # For comparison, use the same time step as Cross scheme
    dt_stability = 1.0 / (a * np.sqrt(1/hx**2 + 1/hy**2))
    J = 2 + int(np.round(T / dt_stability))  # Ensure enough time steps
    dt = T / J
    
    # Создаем массив временных точек
    t = np.zeros(J+1)
    for j in range(J+1):
        t[j] = j * dt
    
    print(f"Factorized scheme: N={N}, M={M}, J={J}")
    print(f"hx={hx:.6f}, hy={hy:.6f}, dt={dt:.6f}")
    
    # Initialize solution array
    w = np.zeros((M+1, N+1, J+1))
    err = np.zeros((M+1, N+1, J+1))  # массив для ошибки
    
    # Set initial condition u(x,y,0)
    for n in range(N+1):
        for m in range(M+1):
            w[m, n, 0] = 5 * np.cos(np.pi * x[n] / 2) * np.cos(3 * np.pi * y[m] / 2)
            # Вычисляем ошибку для начального условия
            err[m, n, 0] = w[m, n, 0] - analytical_solution(x[n], y[m], 0)
    
    # Set initial derivative du/dt(x,y,0) with second-order accuracy
    lambda_mn = (np.pi/2)**2 + (3*np.pi/2)**2
    for n in range(N+1):
        for m in range(M+1):
            # Вычисляем второе приближение с использованием уравнения волны
            w[m, n, 1] = w[m, n, 0] + dt**2/2 * (-25*np.pi**2/2 * np.cos(np.pi * x[n] / 2) * np.cos(3 * np.pi * y[m] / 2) + 2*(x[n] + y[m]))
            # Вычисляем ошибку для первого шага по времени
            err[m, n, 1] = w[m, n, 1] - analytical_solution(x[n], y[m], t[1])
    
    # Time stepping using Factorized scheme
    all_solutions = [w[:, :, 0].copy(), w[:, :, 1].copy()]
    all_errors = [err[:, :, 0].copy(), err[:, :, 1].copy()]
    
    # Setup coefficient constants for the scheme
    Ax = sigma * dt**2 / hx**2
    Cx = 1 + 2 * Ax
    Ay = sigma * dt**2 / hy**2
    Cy = 1 + 2 * Ay
    
    # Preallocate arrays for the tridiagonal solver
    alpha_x = np.zeros(N)
    beta_x = np.zeros(N)
    alpha_y = np.zeros(M)
    beta_y = np.zeros(M)
    
    # Промежуточная функция для факторизованной схемы
    w1 = np.zeros((M+1, N+1))
    
    for j in range(1, J):
        t_next = t[j+1]
        
        # Первый этап: вычисление вспомогательной функции w1
        for m in range(1, M):
            # Прямой ход прогонки по x
            alpha_x[0] = 1
            beta_x[0] = 0
            
            for n in range(1, N):
                # Вычисляем вторые производные
                w_xx = (w[m, n-1, j] - 2*w[m, n, j] + w[m, n+1, j]) / hx**2
                w_yy = (w[m-1, n, j] - 2*w[m, n, j] + w[m+1, n, j]) / hy**2
                
                # Правая часть уравнения
                F = w_xx + w_yy + 2*(x[n] + y[m])
                
                # Коэффициенты прогонки
                alpha_x[n] = Ax / (Cx - Ax * alpha_x[n-1])
                beta_x[n] = (F + Ax * beta_x[n-1]) / (Cx - Ax * alpha_x[n-1])
            
            # Обратный ход прогонки по x
            w1[m, N] = t_next**2 * (1 + y[m])  # Граничное условие при x=1
            
            for n in range(N-1, -1, -1):
                w1[m, n] = alpha_x[n] * w1[m, n+1] + beta_x[n]
        
        # Второй этап: вычисление функции w на новом временном слое
        for n in range(1, N):
            # Прямой ход прогонки по y
            alpha_y[0] = 1
            beta_y[0] = -hy * t_next**2  # Граничное условие при y=0
            
            for m in range(1, M):
                # Правая часть уравнения
                F = dt**2 * w1[m, n] + Ay * (w[m-1, n, j-1] - 2*w[m-1, n, j]) - \
                    Cy * (w[m, n, j-1] - 2*w[m, n, j]) + \
                    Ay * (w[m+1, n, j-1] - 2*w[m+1, n, j])
                
                # Коэффициенты прогонки
                alpha_y[m] = Ay / (Cy - Ay * alpha_y[m-1])
                beta_y[m] = (F + Ay * beta_y[m-1]) / (Cy - Ay * alpha_y[m-1])
            
            # Обратный ход прогонки по y
            w[M, n, j+1] = t_next**2 * (1 + x[n])  # Граничное условие при y=1
            
            for m in range(M-1, -1, -1):
                w[m, n, j+1] = alpha_y[m] * w[m+1, n, j+1] + beta_y[m]
                # Вычисляем ошибку
                err[m, n, j+1] = w[m, n, j+1] - analytical_solution(x[n], y[m], t_next)
        
        # Граничные условия на x=0 и x=1
        for m in range(M+1):
            w[m, 0, j+1] = w[m, 1, j+1] - hx * t_next**2  # Условие Неймана при x=0
            w[m, N, j+1] = t_next**2 * (1 + y[m])         # Условие Дирихле при x=1
            # Вычисляем ошибку для границ
            err[m, 0, j+1] = w[m, 0, j+1] - analytical_solution(x[0], y[m], t_next)
            err[m, N, j+1] = w[m, N, j+1] - analytical_solution(x[N], y[m], t_next)
        
        all_solutions.append(w[:, :, j+1].copy())
        all_errors.append(err[:, :, j+1].copy())
        
        # Выводим максимальную ошибку на каждом временном слое для мониторинга
        if j % 20 == 0:
            print(f"Time step {j}, max error: {np.max(np.abs(err[:, :, j+1])):.6e}")
    
    return w[:, :, J], x, y, dt, J, all_solutions, err[:, :, J]


def plot_solution_2d(solution, x_grid, y_grid, title, analytical=None, error=None):
    """
    Plot a 2D solution as a surface.
    
    Parameters:
    -----------
    solution : ndarray
        2D array of solution values
    x_grid : ndarray
        Grid points in x direction
    y_grid : ndarray
        Grid points in y direction
    title : str
        Plot title
    analytical : ndarray, optional
        Analytical solution for comparison
    error : ndarray, optional
        Precalculated error array
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Plot numerical solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, solution, cmap=cm.viridis, 
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('Numerical Solution')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    if analytical is not None:
        # Plot analytical solution
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, analytical, cmap=cm.viridis,
                                linewidth=0, antialiased=True)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title('Analytical Solution')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # Plot error
        ax3 = fig.add_subplot(133, projection='3d')
        if error is not None:
            error_plot = error
        else:
            error_plot = np.abs(solution - analytical)
        
        surf3 = ax3.plot_surface(X, Y, error_plot, cmap=cm.viridis,
                                linewidth=0, antialiased=True)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('Error')
        ax3.set_title(f'Error (Max = {np.max(np.abs(error_plot)):.6e})')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main function to compare the Cross scheme and Factorized scheme
    for the self-study problem.
    """
    # Parameters
    N = 100     # Число узлов по x
    M = 100     # Число узлов по y
    T = 1.0     # Final time
    sigma = 0.25  # Weight parameter for factorized scheme
    
    # Solve using Cross scheme
    solution_cross, x_grid, y_grid, dt, J, all_cross, error_cross = cross_scheme_2d(N, M, T)
    
    # Solve using Factorized scheme
    solution_factorized, _, _, _, _, all_factorized, error_factorized = factorized_scheme_2d(N, M, T, sigma)
    
    # Compute analytical solution on the grid
    X, Y = np.meshgrid(x_grid, y_grid)
    analytical = analytical_solution(X, Y, T)
    
    # Визуализация для последнего временного слоя
    print("\nПостроение графиков для конечного момента времени t = 1.0")
    fig_cross = plot_solution_2d(solution_cross, x_grid, y_grid, 
                    f'Cross Scheme (N={N}, M={M}, t={T})', analytical, error_cross)
    plt.savefig('self_study_cross_scheme_final.png')
    plt.close(fig_cross)
    
    fig_factorized = plot_solution_2d(solution_factorized, x_grid, y_grid, 
                    f'Factorized Scheme (N={N}, M={M}, σ={sigma}, t={T})', analytical, error_factorized)
    plt.savefig('self_study_factorized_scheme_final.png')
    plt.close(fig_factorized)
    
    # Также для промежуточных моментов времени
    j0 = int(J * 0.7)  # индекс для визуализации промежуточного результата (70% времени)
    if j0 < J:
        print(f"\nПостроение графиков для промежуточного момента времени t = {j0 * dt:.4f}")
        X, Y = np.meshgrid(x_grid, y_grid)
        analytical_mid = analytical_solution(X, Y, j0 * dt)
        
        fig_cross_mid = plot_solution_2d(all_cross[j0], x_grid, y_grid, 
                        f'Cross Scheme (N={N}, M={M}, t={j0*dt:.4f})', analytical_mid)
        plt.savefig('self_study_cross_scheme_mid.png')
        plt.close(fig_cross_mid)
        
        fig_factorized_mid = plot_solution_2d(all_factorized[j0], x_grid, y_grid, 
                        f'Factorized Scheme (N={N}, M={M}, σ={sigma}, t={j0*dt:.4f})', analytical_mid)
        plt.savefig('self_study_factorized_scheme_mid.png')
        plt.close(fig_factorized_mid)
    
    # Calculate and print error metrics
    max_error_cross = np.max(np.abs(error_cross))
    max_error_factorized = np.max(np.abs(error_factorized))
    
    print("\nError comparison at final time:")
    print(f"Cross scheme max error: {max_error_cross:.6e}")
    print(f"Factorized scheme max error: {max_error_factorized:.6e}")
    
    # Reverse the ratio if factorized is better, for consistency
    if max_error_factorized < max_error_cross:
        print(f"Improvement ratio: {max_error_cross / max_error_factorized:.2f}x (factorized scheme is better)")
    else:
        print(f"Improvement ratio: {max_error_factorized / max_error_cross:.2f}x (cross scheme is better)")
    
    print("\nСтатистика по ошибкам:")
    print(f"Cross scheme mean error: {np.mean(np.abs(error_cross)):.6e}")
    print(f"Factorized scheme mean error: {np.mean(np.abs(error_factorized)):.6e}")
    print(f"Cross scheme median error: {np.median(np.abs(error_cross)):.6e}")
    print(f"Factorized scheme median error: {np.median(np.abs(error_factorized)):.6e}")
    
    # График распределения ошибок
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(error_cross).flatten(), bins=50, alpha=0.7)
    plt.title('Cross Scheme Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(error_factorized).flatten(), bins=50, alpha=0.7)
    plt.title('Factorized Scheme Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()


if __name__ == "__main__":
    main() 