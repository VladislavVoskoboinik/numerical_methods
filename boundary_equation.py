import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
t_0 = 0; T = 0.3
a = 0,; b = 1

def u_left(t):
    return np.exp(-t)

def u_init(x):
    return -x + 1

def f(y, t):
    f = np.empty(N)
    f[0] = -y[0] * (y[0]-u_left(t)) / h + np.exp(y[0]**2)
    for n in range(1, N):
        f[n] = -y[n] * (y[n]-y[n-1]) / h + np.exp(y[n]**2)
    return f


def f_y(y, t):
    f_y = np.zeros((N, N))
    f_y[0] = -2*y[0] * (y[0]+u_left(t)) / h + 2*y[0]*np.exp(y[0]**2)
    for n in range(1, N):
        f_y[n, n-1] = y[n]/h
        f_y[n, n] = (-2*y[n]+y[n-1])/h + 2*y[n]*np.exp(y[n]**2)
N = 50
M = 150
x, h = np.linspace(a, b, N+1, retstep = True)
t, tau = np.linspace(t_0, T, M+1, retstep=True)
alpha = (1 + 1j)/2

u = np.empty((M+1, N+1))
u[0] = u_init(x)
y = np.empty((M+1, N))
y[0] = u[0, 1:N+1]



for m in range(0, M):
    w_1 = np.linalg.solve(np.eye(N) - alpha * tau * f_y(y[m], t[m], f(y[m], t[m] + tau/2)))
    y[m+1] = y[m] + tau * w_1.real
    u[m+1, 1:N+1] = y[m+1]
    u[m+1, 0] = u_left(t[m])



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(a, b)
ax.set_ylim(np.min(u), np.max(u))
ax.set_xlabel('x')
ax.set_title('Эволюция решения')
line, = ax.plot([], [], '-or')

# Функция инициализации
def init():
    line.set_data([], [])
    return line,

# Функция обновления кадров
def update(frame):
    line.set_data(x, u[frame])
    ax.set_title(f'Время t = {t[frame]:.2f}')
    return line,

# Создаем анимацию
ani = FuncAnimation(
    fig,
    update,
    frames=len(u),
    init_func=init,
    interval=50,
    blit=True,
    repeat=True
)

plt.show()
"""""
plt.figure(figsize=(10, 6))
selected_times = [0, M//4, M//2, 3*M//4, M]
for step in selected_times:
    plt.plot(x, u[step], label=f't={t[step]:.2f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Решение на разных временных слоях')
plt.legend()
plt.grid(True)
plt.show()

# Анимация
fig, ax = plt.subplots()
line, = ax.plot(x, u[0])
ax.set_xlim(a, b)
ax.set_ylim(np.min(u), np.max(u))
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('Эволюция решения')

def update(frame):
    line.set_ydata(u[frame])
    ax.set_title(f'Время t={t[frame]:.2f}')
    return line,

ani = FuncAnimation(fig, update, frames=M+1, interval=50, blit=True)
plt.show()
"""""