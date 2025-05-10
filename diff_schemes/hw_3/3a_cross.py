from numpy import zeros, cos, sin, pi, meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Схема "крест" для уравнения колебаний на отрезке

# Создание сетки:
N = 50 # число узлов по x
M = 60 # число узлов по t должно удовлетворять условию устойчивости
case_grid = 2 # выбор сетки

x = zeros(N+1)
if case_grid == 1: # сетка, дающая первый порядок аппроксимации по h:
    h = 1/N
    x0 = 0
if case_grid == 2: # сетка, дающая второй порядок аппроксимации по h:
    h = 1/(N-1)
    x0 = -h/2

for i in range(N+1):
    x[i] = x0 + i*h

t = zeros(M+1)
tau = 1/M
for j in range(M+1):
    t[j] = j*tau

# Численное решение:
y = zeros((M+1, N+1))
# Начальное условие:
for i in range(N+1):
    y[0][i] = x[i]**2
    y[1][i] = cos(pi*x[i])*tau + tau**2 + y[0][i]

# Основной цикл по времени:
for j in range(1, M):
    # Внутренние точки:
    for i in range(1, N):
        y[j+1][i] = 2*y[j][i] - y[j-1][i] + tau**2*(y[j][i+1] - 2*y[j][i] + y[j][i-1])/h**2
    # Граничные условия:
    y[j+1][0] = y[j+1][1]
    y[j+1][N] = y[j+1][N-1] + 2*h

# Аналитическое решение и погрешность:
u = zeros((M+1, N+1))
err = zeros((M+1, N+1))
for i in range(N+1):
    for j in range(M+1):
        u[j][i] = t[j]**2 + 1/pi*sin(pi*t[j])*cos(pi*x[i]) + x[i]**2
        err[j][i] = u[j][i] - y[j][i]

# Визуализация в 3D
X, T = meshgrid(x, t)

# Численное решение
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X, T, y, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('y')
ax1.set_title('Численное решение')
fig1.colorbar(surf1)

# Аналитическое решение
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, T, u, cmap='plasma')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u')
ax2.set_title('Аналитическое решение')
fig2.colorbar(surf2)

# Погрешность
fig3 = plt.figure(figsize=(12, 8))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(X, T, err, cmap='coolwarm')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('Ошибка')
ax3.set_title('Погрешность')
fig3.colorbar(surf3)

plt.show()