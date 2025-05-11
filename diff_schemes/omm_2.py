import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

def solve_heat_equation(Nx, Ny, Nt, Lx, Ly, T, D=9):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / Nt
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t = np.linspace(0, T, Nt + 1)
    rx = D * dt / (2 * dx**2)
    ry = D * dt / (2 * dy**2)
    u = np.zeros((Nt + 1, Nx, Ny))
    u_half = np.zeros((Nx, Ny))
    # Начальное условие
    for i in range(Nx):
        for j in range(Ny):
            u[0, i, j] = np.sin(2 * x[i]) * np.sin(3 * y[j])
    # Граничные условия автоматически соблюдаются (u=0 на границе)
    diag_x = (1 + 2*rx) * np.ones(Nx - 2)
    lower_x = -rx * np.ones(Nx - 3)
    upper_x = -rx * np.ones(Nx - 3)
    diag_y = (1 + 2*ry) * np.ones(Ny - 2)
    lower_y = -ry * np.ones(Ny - 3)
    upper_y = -ry * np.ones(Ny - 3)
    for n in range(Nt):
        # Первый полушаг (x - implicit, y - explicit)
        for j in range(1, Ny-1):
            rhs = np.zeros(Nx-2)
            for i in range(1, Nx-1):
                rhs[i-1] = u[n, i, j] + ry * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1])
            ab_x = np.zeros((3, Nx-2))
            ab_x[0,1:] = upper_x
            ab_x[1,:] = diag_x
            ab_x[2,:-1] = lower_x
            u_half[1:Nx-1, j] = solve_banded((1,1), ab_x, rhs)
        # Границы
        u_half[0,:] = u_half[-1,:] = u_half[:,0] = u_half[:,-1] = 0
        # Второй полушаг (y - implicit, x - explicit)
        for i in range(1, Nx-1):
            rhs = np.zeros(Ny-2)
            for j in range(1, Ny-1):
                rhs[j-1] = u_half[i, j] + rx * (u_half[i+1, j] - 2*u_half[i, j] + u_half[i-1, j])
            ab_y = np.zeros((3, Ny-2))
            ab_y[0,1:] = upper_y
            ab_y[1,:] = diag_y
            ab_y[2,:-1] = lower_y
            u[n+1, i, 1:Ny-1] = solve_banded((1,1), ab_y, rhs)
        u[n+1, 0,:] = u[n+1, -1,:] = u[n+1, :,0] = u[n+1, :,-1] = 0
    return u, x, y, t

def analytical_solution(x, y, t, D=9):
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.zeros((len(t), len(x), len(y)))
    for n in range(len(t)):
        u_exact[n] = np.sin(2*X) * np.sin(3*Y) * np.exp(-D*(2**2 + 3**2)*t[n])
    return u_exact

# Параметры задачи
Nx, Ny, Nt = 21, 41, 100
Lx, Ly, T = np.pi/2, np.pi, 0.1

# Решение
u_num, x, y, t = solve_heat_equation(Nx, Ny, Nt, Lx, Ly, T)
u_an = analytical_solution(x, y, t)

# Визуализация для нескольких моментов времени
fig = plt.figure(figsize=(12,8))
times_to_plot = [0, Nt//4, Nt//2, Nt]
for k, n in enumerate(times_to_plot):
    ax = fig.add_subplot(2, 2, k+1, projection='3d')
    X, Y = np.meshgrid(x, y, indexing='ij')
    ax.plot_surface(X, Y, u_num[n], cmap='viridis', alpha=0.8, label='Численное')
    ax.plot_wireframe(X, Y, u_an[n], color='red', linewidth=0.5, alpha=0.8, label='Аналитическое')
    ax.set_title(f't = {t[n]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
plt.tight_layout()
plt.show()

# Сравнение численного и аналитического решений в центре области
plt.figure(figsize=(8,5))
plt.plot(t, [u_num[n, Nx//2, Ny//2] for n in range(Nt+1)], label='Численное')
plt.plot(t, [u_an[n, Nx//2, Ny//2] for n in range(Nt+1)], '--', label='Аналитическое')
plt.xlabel('t')
plt.ylabel('u(x=π/4, y=π/2, t)')
plt.legend()
plt.title('Сравнение численного и аналитического решений в центре области')
plt.grid(True)
plt.show()
