# -*- coding: utf-8 -*-
from numpy import zeros, linspace, pi, sqrt, linalg
from matplotlib import pyplot as plt
from celluloid import Camera


def f(u, g, mass_1, mass_2, l):
    f = zeros(11)
    f[0] = u[4]
    f[1] = u[5]
    f[2] = u[6]
    f[3] = u[7]
    f[4] = 2 / mass_1 * u[10] * (u[0] - u[2])
    f[5] = -g + 1 / mass_1 * u[8] + 2 / mass_1 * u[10] * (u[1] - u[3])
    f[6] = u[9] - 2 / mass_2 * u[10] * (u[0] - u[2])
    f[7] = -g - 2 / mass_2 * u[10] * (u[1] - u[3])
    f[8] = u[1]
    f[9] = u[2]
    f[10] = (u[0] - u[2])**2 + (u[1] - u[3])**2 - l**2
    return f


def D():
    D = zeros((11, 11))
    for i in range(8):
        D[i, i] = 1.
    return D


def f_u(u, mass_1, mass_2):
    f_u = zeros((11, 11))
    f_u[0, 4] = 1
    f_u[1, 5] = 1
    f_u[2, 6] = 1
    f_u[3, 7] = 1

    f_u[4, 0] = 2 * u[10] / mass_1
    f_u[4, 2] = -2 * u[10] / mass_1
    f_u[4, 10] = 2 * (u[0] - u[2]) / mass_1

    f_u[5, 1] = 2 * u[10] / mass_1
    f_u[5, 3] = -2 * u[10] / mass_1
    f_u[5, 8] = 1 / mass_1
    f_u[5, 10] = 2 * (u[1] - u[3]) / mass_1

    f_u[6, 0] = -2 * u[10] / mass_2
    f_u[6, 2] = 2 * u[10] / mass_2
    f_u[6, 9] = 1
    f_u[6, 10] = -2 * (u[0] - u[2]) / mass_2

    f_u[7, 1] = -2 * u[10] / mass_2
    f_u[7, 3] = 2 * u[10] / mass_2
    f_u[7, 10] = -2 * (u[1] - u[3]) / mass_2

    f_u[8, 1] = 1
    f_u[9, 2] = 1

    f_u[10, 0] = 2 * (u[0] - u[2])
    f_u[10, 1] = 2 * (u[1] - u[3])
    f_u[10, 2] = -2 * (u[0] - u[2])
    f_u[10, 3] = -2 * (u[1] - u[3])
    return f_u


t_0 = 0.
T = 10
l = 5.
x_1_0 = 4.
y_1_0 = 0.
x_2_0 = 0.
y_2_0 = sqrt(l**2 - (x_1_0 - x_2_0)**2)
v_x_1_0 = 0.
v_y_1_0 = 0.
v_x_2_0 = 0.
v_y_2_0 = 0.
g = 1
mass_1 = 1
mass_2 = 20
lambda_1_0 = 100
lambda_2_0 = 100
lambda_3_0 = 100

alpha = (1 + 1j) / 2
M = 500
tau = (T - t_0) / M
t = linspace(t_0, T, M + 1)
u = zeros((M + 1, 11))
u[0, :] = [x_1_0, y_1_0, x_2_0, y_2_0, v_x_1_0, v_y_1_0, v_x_2_0, v_y_2_0, lambda_1_0, lambda_2_0, lambda_3_0]

# Метод ROS1
fall_complete = False
for m in range(M):
    if fall_complete:
        u[m+1] = u[m]
        continue
        
    w_1 = linalg.solve(D() - alpha * tau * f_u(u[m], mass_1, mass_2), f(u[m], g, mass_1, mass_2, l))
    u[m + 1] = u[m] + tau * w_1.real
    
    if u[m+1, 1] <= 0 and u[m+1, 3] <= 0:
        u[m+1, 1] = 0
        u[m+1, 3] = 0
        fall_complete = True


fig, ax = plt.subplots(figsize=(10, 8))
camera = Camera(fig)

# Найдем момент, когда лестница полностью упала
fall_frame = M
for m in range(M + 1):
    if u[m, 1] <= 0 and u[m, 3] <= 0:
        fall_frame = m
        break

for m in range(0, M + 1, 2):
    x1, y1 = max(u[m, 0], 0), max(u[m, 1], 0)
    x2, y2 = max(u[m, 2], 0), max(u[m, 3], 0)
    
    ax.plot([x1, x2], [y1, y2], 'b-', lw=3)
    ax.plot(x1, y1, 'ro', markersize=8)
    ax.plot(x2, y2, 'go', markersize=8)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    camera.snap()

"""""
for _ in range(10):
    m = min(fall_frame, M)
    x1, y1 = max(u[m, 0], 0), max(u[m, 1], 0)
    x2, y2 = max(u[m, 2], 0), max(u[m, 3], 0)

    ax.plot([x1, x2], [y1, y2], 'b-', lw=3)
    ax.plot(x1, y1, 'ro', markersize=8)
    ax.plot(x2, y2, 'go', markersize=8)
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.set_xlabel("X позиция")
    ax.set_ylabel("Y позиция")
    
    camera.snap()
"""""
animation = camera.animate(interval=1)
plt.show()
