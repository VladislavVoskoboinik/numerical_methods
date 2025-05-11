import numpy as np
import math

# Начальные условия
x_0 = np.pi / 2
y_0 = 1
v_0 = 0
w_0 = 1
X = 10
r = 2      # шаг сгущения
S = 6      # уровни сгущения
p = 3      # порядок схемы по пространству
q = 1      # шаг сгущения
eps = 1e-2

def f(u, x):
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[0] - 1) * np.cos(x) / np.sin(x)
    return f

def ERK3_step(current_x, current_u, current_step):
    w1 = f(current_u, current_x)
    w2 = f(current_u + 0.5 * current_step * w1, current_x + 0.5 * current_step)
    w3 = f(current_u + 0.75 * current_step * w2, current_x + 0.75 * current_step)
    return current_u + current_step * (2/9*w1 + 3/9*w2 + 4/9*w3)

def ERK3_local_thickening(M, eps, x_0, X):
    tau = (X - x_0) / M
    u = [[y_0, v_0, w_0]]
    u_thickened = [[y_0, v_0, w_0]]
    x = [x_0]
    m = 0

    while x[-1] < X and m <= 5 * M:
        u_next = ERK3_step(x[-1], u[-1], tau)
        ut1 = ERK3_step(x[-1], u_thickened[-1], tau / r)
        ut2 = ERK3_step(x[-1] + tau / r, ut1, tau / r)

        error = np.linalg.norm(u_next - ut2)
        if x[-1] + tau > X:
            tau = X - x[-1]
        u.append(u_next)
        u_thickened.append(ut1)
        u_thickened.append(ut2)
        x.append(x[-1] + tau)

        if error > eps:
            tau_new = ((eps * (r ** p - 1) * tau ** (p + 1)) / (error * (X - x_0))) ** (1 / p)
            tau = max(tau_new, tau / 2)
        else:
            tau = min(2 * tau, (X - x_0) / M)
        m += 1

    return np.array(x), np.array(u)


U = np.zeros((S, S))
R = np.zeros((S, S))
P = np.zeros((S, S))

for s in range(S):
    M_s = r ** s * 10
    _, u_vals = ERK3_local_thickening(M_s, eps, x_0, X)
    U[s, 0] = u_vals[len(u_vals)//2, 0]

for s in range(1, S):
    for l in range(s):
        R[s, l] = (U[s, l] - U[s-1, l]) / (r**(p + l*q) - 1)
        U[s, l+1] = U[s, l] + R[s, l]

for s in range(2, S):
    for l in range(s-1):
        num = np.abs(U[s-1, l] - U[s-2, l])
        den = np.abs(U[s, l] - U[s-1, l])
        if den != 0:
            P[s, l] = np.log(num / den) / np.log(r)
        else:
            P[s, l] = np.nan


def PrintTriangular(A, i, title):
    print(title)
    print(' ', end=' ')
    for l in range(len(A)):
        print(f'p={p + l*q:<4d}', end=' ')
    print()
    for m in range(len(A)):
        print(f's={m:<2d}', end=' ')
        for l in range(m + 1 - i):
            print(f'{A[m, l]:9.6f}', end=' ')
        print()
    print()

def PrintTriangularP(A, i, title):
    print(title)
    print(' ', end=' ')
    for l in range(len(A)):
        print(f'p={p + l*q:<4d}', end=' ')
    print()
    for m in range(len(A)):
        print(f's={m:<2d}', end=' ')
        for l in range(m + 1 - i):
            if np.isnan(A[m, l]):
                print(f"{'—':>9}", end=' ')
            else:
                print(f'{A[m, l]:9.4f}', end=' ')
        print()
    print()

PrintTriangular(U, 0, 'Таблица приближённых значений y(X):')
PrintTriangular(R, 1, 'Таблица оценок ошибок:')
PrintTriangularP(P, 2, 'Таблица эффективного порядка точности:')
