import numpy as np
import matplotlib.pyplot as plt


# Явная схема для уравнения теплопроводности на отрезке

# Создание сетки ----------------------------------------------------------
N = 10                      # число интервалов по x
M = 10 + 2 * (N ** 2)       # число интервалов по t (M >= 2 * N^2)

L = 2
h = L / N
tau = 1 / M

x = np.linspace(0, L, N + 1)
t = np.linspace(0, 1, M + 1)
# -------------------------------------------------------------------------

# Численное решение -------------------------------------------------------
y = np.zeros((M + 1, N + 1))
case_bound = 1  # выбор варианта граничного условия

# Начальное условие:
for i in range(N + 1):
    y[0, i] = 3 - x[i] + np.cos(3 * x[i] * np.pi / 4)

# Явная схема:
for j in range(M):
    # Внутренние точки:
    for i in range(1, N):
        y[j + 1, i] = y[j, i] + tau * (y[j, i + 1] - 2 * y[j, i] + y[j, i - 1]) / (h * h)

    # Граничные условия:
    y[j + 1, N] = 1 

    # Первый вариант граничного условия при x = 0:
    if case_bound == 1:
        y[j + 1, 0] = y[j + 1, 1] + h*t[j+1]

    # Альтернативные граничные условия:
    '''
    # Второй вариант:
    if case_bound == 2:
        y[j + 1, N] = (
            y[j + 1, N - 1] + h * t[j + 1] + h ** 2 * y[j, N] / (2 * tau) + h ** 2 / 2
        ) / (1 + h ** 2 / (2 * tau))

    # Третий вариант:
    if case_bound == 3:
        y[j + 1, N] = (4 * y[j + 1, N - 1] - y[j + 1, N - 2]) / 3 + 2 * h * t[j + 1] / 3
    '''
# -------------------------------------------------------------------------

# Аналитическое решение и погрешность -------------------------------------
u = np.zeros((M + 1, N + 1))
err = np.zeros((M + 1, N + 1))

def u_a(x_val, t_val):

    return 3 - x_val + np.cos(3 * np.pi * x_val / 4) * np.exp(-(3 * np.pi / 4)**2 * t_val)

for i in range(N + 1):
    for j in range(M + 1):
        u[j, i] = u_a(x[i], t[j])
        err[j, i] = u[j, i] - y[j, i]

# Норма погрешности на последнем слое:
max_err = np.max(np.abs(err[M]))
print("Максимальная погрешность на последнем слое:", max_err)
# -------------------------------------------------------------------------

# Визуализация ------------------------------------------------------------
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

plt.show()
# -------------------------------------------------------------------------
