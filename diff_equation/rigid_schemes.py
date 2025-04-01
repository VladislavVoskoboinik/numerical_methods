from numpy import pi, empty, zeros, linspace, cos, sin, sqrt
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


t_0 = 0
T = 2
u_0 = 1
M = 150
tau = (T - t_0) / M


def f(u):
    f = empty(1)
    f[0] = u[0]**2
    return f

# Массивы для хранения результатов
u_euler = empty((M + 1, 1))

# Устанавливаем начальные условия
u_euler[0] = u_0


t, tau = linspace(t_0, T, M + 1, retstep=True)
for m in range(M):
    x = u_euler[m]
    S = 1
    for s in range(S):
        x = x - (x - u_euler[m] - tau * x ** 2)/(1 - 2*tau*x)
    u_euler[m+1] = x


# Теоретическое решение u(t) = 1/(1-t)
u_theory = 1/(1-t)


# Построение графика u(t)
plt.figure(figsize=(10, 6))
plt.xlim(0, 2)
plt.ylim(-30, 30)
plt.plot(t, u_euler[:, 0], "-r", label="Схема Эйлера", marker="o", markersize=4)
plt.plot(t, u_theory, "--k", label="Теоретическое решение")
# plt.plot(t, u_rk2[:, 0], "-b", label="Схема Рунге-Кутта") 
plt.title("График u(t)")
plt.xlabel("Время t (с)")
plt.ylabel("u")
plt.grid(True)
plt.legend()
plt.show()
