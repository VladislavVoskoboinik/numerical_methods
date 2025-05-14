import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Неявная схема для уравнения теплопроводности на отрезке

# ------------------------- Параметры задачи -------------------------
N = 20                 # Число интервалов по x
M = 100                # Число интервалов по t (M >= 2 * N^2)
case_bound = 1         # Вариант граничного условия
L = 2                  # Длина отрезка по x
T_max = 1              # Максимальное время

h = L / N              # Шаг по x
tau = T_max / M        # Шаг по времени

x = np.linspace(0, L, N + 1)
t = np.linspace(0, T_max, M + 1)
# --------------------------------------------------------------------

# --------------------- Инициализация решения ------------------------
y = np.zeros((M + 1, N + 1))  # Численное решение

# Начальное условие
y[0] = 3 - x + np.cos(3 * x * np.pi / 4)

# Прогоночные коэффициенты
alpha = np.zeros(N)
beta = np.zeros(N)

A = tau / h**2
B = A
C = 2 * A + 1
# --------------------------------------------------------------------

# ------------------------- Прогонка (неявная схема) -----------------
for j in range(M):
    # Прямой ход
    alpha[0] = 1
    beta[0] = h  # Граничное условие на левом краю

    for i in range(1, N):
        F = y[j][i]
        denom = C - A * alpha[i - 1]
        alpha[i] = B / denom
        beta[i] = (A * beta[i - 1] + F) / denom

    # Граничное условие справа (вариант 1)
    if case_bound == 1:
        kappa_2 = 0
        mu_2 = 1

    # Обратный ход
    y[j + 1][N] = (mu_2 + kappa_2 * beta[N - 1]) / (1 - kappa_2 * alpha[N - 1])

    for i in reversed(range(N)):
        y[j + 1][i] = alpha[i] * y[j + 1][i + 1] + beta[i]
# --------------------------------------------------------------------

# -------------------- Аналитическое решение и ошибка ----------------
u = np.zeros((M + 1, N + 1))     # Аналитическое решение
err = np.zeros((M + 1, N + 1))   # Погрешность

def u_a(x_val, t_val, _):
    """Аналитическое решение"""
    return 3 - x_val + np.cos(3 * np.pi * x_val / 4) * np.exp(-(3 * np.pi / 4)**2 * t_val)

# Заполнение аналитического решения и ошибок
for i in range(N + 1):
    for j in range(M + 1):
        u[j][i] = u_a(x[i], t[j], 0)
        err[j][i] = u[j][i] - y[j][i]

# Норма ошибки на последнем временном слое
max_err = np.max(np.abs(err[M]))
print("Максимальная погрешность на последнем слое:", max_err)
# --------------------------------------------------------------------

# ----------------------------- Визуализация --------------------------
X, T = np.meshgrid(x, t)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, T, u, cmap='inferno')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u')
ax1.set_title('Аналитическое решение')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, T, y, cmap='inferno')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('y')
ax2.set_title('Численное решение')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, T, err, cmap='inferno')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('Ошибка')
ax3.set_title('Погрешность')

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------
