import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def analytical_solution(x, y, t):
    """
    Analytical solution for the example problem:
    u = txy + cos(π√2t) * sin(πx/2) * sin(πy/2)
    """
    return t * x * y + np.cos(np.pi * np.sqrt(2) * t) * np.sin(np.pi * x / 2) * np.sin(np.pi * y / 2)


def cross_scheme_2d(N, M, T, a=2.0):
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
        Wave speed coefficient (default=2.0 for a²=4)
    
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
    # Grid spacing
    hx = 2.0 / N
    hy = 1.0 / (M - 0.5)
    
    # Create grid points
    x = np.linspace(0, 2, N+1)  # [0, 2]
    y = np.linspace(0, 1 + hy/2, M+1)  # [0, 1 + hy/2] to handle Neumann boundary at y=1
    
    # Compute time step for stability
    dt_stability = 1.0 / (a * np.sqrt(1/hx**2 + 1/hy**2))
    J = 1 + int(np.ceil(T / dt_stability))
    dt = T / J
    
    print(f"Cross scheme: N={N}, M={M}, J={J}")
    print(f"hx={hx:.6f}, hy={hy:.6f}, dt={dt:.6f}")
    
    # Initialize solution array
    w = np.zeros((J+1, N+1, M+1))
    
    # Set initial condition u(x,y,0)
    for n in range(N+1):
        for m in range(M+1):
            w[0, n, m] = np.sin(np.pi * x[n] / 2) * np.sin(np.pi * y[m] / 2)
    
    # Set initial derivative du/dt(x,y,0) with second-order accuracy
    for n in range(N+1):
        for m in range(M+1):
            w[1, n, m] = w[0, n, m] + dt * (x[n] * y[m]) + dt**2/2 * (-np.pi**2) * np.sin(np.pi * x[n] / 2) * np.sin(np.pi * y[m] / 2)
    
    # Apply boundary conditions for t=0 and t=dt
    for j in range(2):
        t = j * dt
        # Boundary at x=0
        w[j, 0, :] = 0
        # Boundary at x=2
        w[j, N, :] = 2 * t * y
        # Boundary at y=0
        w[j, :, 0] = 0
        # Boundary at y=1 (Neumann condition)
        w[j, :, M] = w[j, :, M-1] + hy * t * x
    
    # Time stepping using Cross scheme
    all_solutions = [w[0].copy(), w[1].copy()]
    
    for j in range(1, J):
        # Interior points
        for n in range(1, N):
            for m in range(1, M):
                w[j+1, n, m] = (
                    2 * w[j, n, m] - w[j-1, n, m] + 
                    4 * dt**2 * (
                        (w[j, n+1, m] - 2*w[j, n, m] + w[j, n-1, m]) / hx**2 +
                        (w[j, n, m+1] - 2*w[j, n, m] + w[j, n, m-1]) / hy**2
                    )
                )
        
        # Apply boundary conditions
        t = (j+1) * dt
        # Boundary at x=0
        w[j+1, 0, :] = 0
        # Boundary at x=2
        w[j+1, N, :] = 2 * t * y
        # Boundary at y=0
        w[j+1, :, 0] = 0
        # Boundary at y=1 (Neumann condition)
        w[j+1, :, M] = w[j+1, :, M-1] + hy * t * x
        
        all_solutions.append(w[j+1].copy())
    
    return w[J], x, y, dt, J, all_solutions


def factorized_scheme_2d(N, M, T, sigma=0.25, a=2.0):
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
        Wave speed coefficient (default=2.0 for a²=4)
    
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
    # Grid spacing
    hx = 2.0 / N
    hy = 1.0 / (M - 0.5)
    
    # Create grid points
    x = np.linspace(0, 2, N+1)  # [0, 2]
    y = np.linspace(0, 1 + hy/2, M+1)  # [0, 1 + hy/2] to handle Neumann boundary
    
    # For comparison, use the same time step as Cross scheme
    dt_stability = 1.0 / (a * np.sqrt(1/hx**2 + 1/hy**2))
    J = 1 + int(np.ceil(T / dt_stability))
    dt = T / J
    
    print(f"Factorized scheme: N={N}, M={M}, J={J}")
    print(f"hx={hx:.6f}, hy={hy:.6f}, dt={dt:.6f}")
    
    # Initialize solution array
    w = np.zeros((J+1, N+1, M+1))
    
    # Set initial condition u(x,y,0)
    for n in range(N+1):
        for m in range(M+1):
            w[0, n, m] = np.sin(np.pi * x[n] / 2) * np.sin(np.pi * y[m] / 2)
    
    # Set initial derivative du/dt(x,y,0) with second-order accuracy
    for n in range(N+1):
        for m in range(M+1):
            w[1, n, m] = w[0, n, m] + dt * (x[n] * y[m]) + dt**2/2 * (-np.pi**2) * np.sin(np.pi * x[n] / 2) * np.sin(np.pi * y[m] / 2)
    
    # Apply boundary conditions for t=0 and t=dt
    for j in range(2):
        t = j * dt
        # Boundary at x=0
        w[j, 0, :] = 0
        # Boundary at x=2
        w[j, N, :] = 2 * t * y
        # Boundary at y=0
        w[j, :, 0] = 0
        # Boundary at y=1 (Neumann condition)
        w[j, :, M] = w[j, :, M-1] + hy * t * x
    
    # Time stepping using Factorized scheme
    all_solutions = [w[0].copy(), w[1].copy()]
    
    # Setup coefficient constants for the scheme
    Ax = 4 * sigma * dt**2 / hx**2
    Cx = 1 + 2 * Ax
    Ay = 4 * sigma * dt**2 / hy**2
    Cy = 1 + 2 * Ay
    
    # Preallocate arrays for the tridiagonal solver
    alpha_x = np.zeros(N)
    beta_x = np.zeros(N)
    alpha_y = np.zeros(M)
    beta_y = np.zeros(M)
    
    for j in range(1, J):
        # Intermediate step - compute auxiliary function w_intermediate
        w_intermediate = np.zeros((N+1, M+1))
        
        # For each fixed m, solve tridiagonal system in x-direction
        for m in range(1, M):
            # Forward sweep of tridiagonal algorithm
            alpha_x[0] = 0
            beta_x[0] = 0
            
            for n in range(1, N):
                # Calculate right-hand side
                v_xx = (w[j, n-1, m] - 2*w[j, n, m] + w[j, n+1, m]) / hx**2
                v_yy = (w[j, n, m-1] - 2*w[j, n, m] + w[j, n, m+1]) / hy**2
                F = 4 * (v_xx + v_yy)
                
                # Calculate next alpha and beta
                alpha_x[n] = Ax / (Cx - Ax * alpha_x[n-1])
                beta_x[n] = (F + Ax * beta_x[n-1]) / (Cx - Ax * alpha_x[n-1])
            
            # Backward substitution
            w_intermediate[N, m] = 0  # boundary condition
            
            for n in range(N-1, 0, -1):
                w_intermediate[n, m] = alpha_x[n] * w_intermediate[n+1, m] + beta_x[n]
            
            # Boundary condition at x=0
            w_intermediate[0, m] = 0
        
        # Boundary conditions for intermediate function
        for m in range(M+1):
            w_intermediate[N, m] = 0  # x=2 (will be set at the end)
            w_intermediate[0, m] = 0  # x=0
        
        # Final step - solve for w at the next time step
        for n in range(1, N):
            # Forward sweep of tridiagonal algorithm in y-direction
            alpha_y[0] = 0
            beta_y[0] = 0
            
            for m in range(1, M):
                # Calculate right-hand side
                F = (dt**2 * w_intermediate[n, m] + 
                     Ay * (w[j-1, n, m-1] - 2*w[j, n, m-1]) - 
                     Cy * (w[j-1, n, m] - 2*w[j, n, m]) + 
                     Ay * (w[j-1, n, m+1] - 2*w[j, n, m+1]))
                
                # Calculate next alpha and beta
                alpha_y[m] = Ay / (Cy - Ay * alpha_y[m-1])
                beta_y[m] = (F + Ay * beta_y[m-1]) / (Cy - Ay * alpha_y[m-1])
            
            # Backward substitution for y-direction
            t = (j+1) * dt
            
            # Special handling for Neumann boundary at y=1 (m=M)
            w[j+1, n, M] = (beta_y[M-1] + hy * t * x[n]) / (1 - alpha_y[M-1])
            
            for m in range(M-1, 0, -1):
                w[j+1, n, m] = alpha_y[m] * w[j+1, n, m+1] + beta_y[m]
            
            # Boundary condition at y=0
            w[j+1, n, 0] = 0
        
        # Apply boundary conditions at x=0 and x=2
        t = (j+1) * dt
        for m in range(M+1):
            w[j+1, 0, m] = 0             # x=0
            w[j+1, N, m] = 2 * t * y[m]  # x=2
        
        all_solutions.append(w[j+1].copy())
    
    return w[J], x, y, dt, J, all_solutions


def plot_solution_2d(solution, x_grid, y_grid, title, analytical=None):
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
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
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
        error = np.abs(solution - analytical)
        surf3 = ax3.plot_surface(X, Y, error, cmap=cm.viridis,
                                linewidth=0, antialiased=True)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('Error')
        ax3.set_title(f'Error (Max = {np.max(error):.6f})')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main function to compare the Cross scheme and Factorized scheme.
    """
    # Parameters
    N = 100    # Grid points in x
    M = 50     # Grid points in y
    T = 1.0    # Final time
    sigma = 0.25  # Weight parameter for factorized scheme
    
    # Solve using Cross scheme
    solution_cross, x_grid, y_grid, dt, J, all_cross = cross_scheme_2d(N, M, T)
    
    # Solve using Factorized scheme
    solution_factorized, _, _, _, _, all_factorized = factorized_scheme_2d(N, M, T, sigma)
    
    # Compute analytical solution on the grid
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    analytical = analytical_solution(X, Y, T)
    
    # Plot solutions at final time
    plot_solution_2d(solution_cross, x_grid, y_grid, 
                    f'Cross Scheme (N={N}, M={M}, t={T})', analytical)
    plt.savefig('cross_scheme_final.png')
    
    plot_solution_2d(solution_factorized, x_grid, y_grid, 
                    f'Factorized Scheme (N={N}, M={M}, σ={sigma}, t={T})', analytical)
    plt.savefig('factorized_scheme_final.png')
    
    # Also plot intermediate time steps
    mid_index1 = int(J/3)
    mid_time1 = mid_index1 * dt
    
    mid_index2 = int(2*J/3)
    mid_time2 = mid_index2 * dt
    
    # Compute analytical solutions at intermediate times
    analytical_mid1 = analytical_solution(X, Y, mid_time1)
    analytical_mid2 = analytical_solution(X, Y, mid_time2)
    
    # Plot results at intermediate times
    plot_solution_2d(all_cross[mid_index1], x_grid, y_grid, 
                    f'Cross Scheme (N={N}, M={M}, t={mid_time1:.4f})', analytical_mid1)
    plt.savefig('cross_scheme_t1.png')
    
    plot_solution_2d(all_cross[mid_index2], x_grid, y_grid, 
                    f'Cross Scheme (N={N}, M={M}, t={mid_time2:.4f})', analytical_mid2)
    plt.savefig('cross_scheme_t2.png')
    
    plot_solution_2d(all_factorized[mid_index1], x_grid, y_grid, 
                    f'Factorized Scheme (N={N}, M={M}, σ={sigma}, t={mid_time1:.4f})', analytical_mid1)
    plt.savefig('factorized_scheme_t1.png')
    
    plot_solution_2d(all_factorized[mid_index2], x_grid, y_grid, 
                    f'Factorized Scheme (N={N}, M={M}, σ={sigma}, t={mid_time2:.4f})', analytical_mid2)
    plt.savefig('factorized_scheme_t2.png')
    
    # Calculate and print error metrics
    error_cross = np.max(np.abs(solution_cross - analytical))
    error_factorized = np.max(np.abs(solution_factorized - analytical))
    
    print("\nError comparison at final time:")
    print(f"Cross scheme max error: {error_cross:.6e}")
    print(f"Factorized scheme max error: {error_factorized:.6e}")
    print(f"Improvement ratio: {error_cross / error_factorized:.2f}x")


if __name__ == "__main__":
    main() 