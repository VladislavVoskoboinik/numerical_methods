from numpy import zeros, cos, sin, pi, meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Неявная схема для уравнения колебаний на отрезке

# Создание сетки в зависимости:
case_sigma = 2
if case_sigma == 1:
    sigma = 0.3
    N = 50
    M = 30
if case_sigma == 3:
    sigma = 0.05
    N = 50
    M = 30
case_grid = 2

if case_grid == 1:
    if case_sigma == 2:
        sigma = 0.05
        N = 50
        M = 55
    h = 1/N
    x0 = 0
if case_grid == 2:
    if case_sigma == 2:
        sigma = 0.05
        N = 50
        M = 54
    h = 1/(N-1)
    x0 = -h/2

x = zeros(N+1)
for i in range(N+1):  # ИСПРАВЛЕНО: заменен while на for
    x[i] = x0 + i*h

t = zeros(M+1)
tau = 1/M
for j in range(M+1):  # ИСПРАВЛЕНО: заменен while на for
    t[j] = j*tau

# Численное решение:
y = zeros((M+1, N+1))
# Начальное условие:
for i in range(N+1):
    y[0][i] = x[i]**2
    y[1][i] = cos(pi*x[i])*tau + tau**2 + y[0][i]

# Прогоночные коэффициенты:
alpha = zeros(N+1)  # ИСПРАВЛЕНО: размер массива
beta = zeros(N+1)   # ИСПРАВЛЕНО: размер массива
kappa = [1, 1]
mu = [0, 2*h]

# Основной цикл по времени:
for j in range(1, M):  # ИСПРАВЛЕНО: заменен while на for
    # Прямой ход прогонки
    alpha[0] = kappa[0]
    beta[0] = mu[0]
    
    for i in range(1, N):  # ИСПРАВЛЕНО: правильный диапазон
        y_xx_j_min1 = (y[j-1][i-1] - 2*y[j-1][i] + y[j-1][i+1])/h**2
        y_xx_j = (y[j][i-1] - 2*y[j][i] + y[j][i+1])/h**2
        F = 2*y[j][i] - y[j-1][i] + tau**2*(sigma*y_xx_j_min1 + (1 - 2*sigma)*y_xx_j)
        
        A = sigma*tau**2/h**2
        B = A
        C = 1 + 2*A
        
        alpha[i] = B/(C - A*alpha[i-1])
        beta[i] = (A*beta[i-1] + F)/(C - A*alpha[i-1])

    # Обратный ход прогонки
    y[j+1][N] = (mu[1] + kappa[1]*beta[N-1])/(1 - kappa[1]*alpha[N-1])  # ИСПРАВЛЕНО: индексы
    
    for i in reversed(range(N)):  # ИСПРАВЛЕНО: правильный обратный ход
        y[j+1][i] = alpha[i]*y[j+1][i+1] + beta[i]
    
    # Явное граничное условие справа  # ИСПРАВЛЕНО: добавлено
    #y[j+1][N] = y[j+1][N-1] + 2*h

# Аналитическое решение и погрешность
u = zeros((M+1, N+1))
err = zeros((M+1, N+1))
for i in range(N+1):
    for j in range(M+1):
        u[j][i] = t[j]**2 + 1/pi*sin(pi*t[j])*cos(pi*x[i]) + x[i]**2
        err[j][i] = u[j][i] - y[j][i]

# 3D визуализация  # ИСПРАВЛЕНО: заменено на 3D графики
X, T = meshgrid(x, t)

fig = plt.figure(figsize=(18, 6))

# Численное решение
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, T, y, cmap='viridis')
ax1.set_title('Численное решение')

# Аналитическое решение
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, T, u, cmap='plasma')
ax2.set_title('Аналитическое решение')

# Погрешность
ax3 = fig.add_subplot(133, projection='3d')
surf = ax3.plot_surface(X, T, err, cmap='coolwarm')
fig.colorbar(surf, ax=ax3)
ax3.set_title('Погрешность')

plt.tight_layout()
plt.show()