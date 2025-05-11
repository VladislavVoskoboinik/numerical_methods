import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

a = 0.0; b = 1.0
t_0 = 0.0; T = 30
lamb = -1
alpha = (1 + 1j)/2  # CROS1 схема
M = 25

def solve(N):
    h = (b - a)/N
    x = np.linspace(a, b, N+1)
    tau = (T - t_0)/M
    t = np.linspace(t_0, T, M+1)
    u = np.zeros((M+1, N+1))
    y = np.zeros((M+1, 3*N-3))

    def u_init(x): return 0
    for n in range(N+1): u[0, n] = u_init(x[n])
    y[0, :N] = 0
    y[0, N:2*N] = -lamb
    y[0, 2*N:3*N-3] = -1

    def f(y, h, N, lamb):
        f_vec = np.zeros(3*N-3)
        f_vec[0] = -y[2*N-2] + (1/h**2)*(y[1] - 2*y[0]) - np.exp(lamb*y[0])
        f_vec[N-2] = -y[3*N-4] + (1/h**2)*(y[N-3] - 2*y[N-2]) - np.exp(lamb*y[N-2])
        for n in range(1, N-2):
            f_vec[n] = -y[n+2*N-2] + (1/h**2)*(y[n+1] - 2*y[n] + y[n-1]) - np.exp(lamb*y[n])
        for n in range(N-1, 2*N-2):
            f_vec[n] = -1 * y[n-N+1]
        for n in range(2*N-2, 3*N-3):
            f_vec[n] = -1 * y[n-N+1]
        return f_vec

    def D(N, h):
        D_mat = np.zeros((3*N-3, 3*N-3))
        for n in range(N-1, 3*N-3):
            D_mat[n, n] = 1
        return D_mat

    def f_y(y, h, N, lamb):
        jac = np.zeros((3*N-3, 3*N-3))
        for n in range(N-1, 2*N-2):
            jac[n, n-N+1] = -1
        for n in range(2*N-2, 3*N-3):
            jac[n, n-N+1] = 1
        jac[0, 1] = 1/h**2
        jac[0, 2*N-2] = -1
        jac[0, 0] = -2/h**2 - lamb*np.exp(lamb*y[0])
        jac[N-2, N-3] = 1/h**2
        jac[N-2, 3*N-4] = -1
        jac[N-2, N-2] = -2/h**2 - lamb*np.exp(lamb*y[N-2])
        for n in range(1, N-2):
            jac[n, n+1] = 1/h**2
            jac[n, n+2*N-2] = -1
            jac[n, n-1] = 1/h**2
            jac[n, n] = -2/h**2 - lamb*np.exp(lamb*y[n])
        return jac

    for m in range(M):
        D_mat = D(N, h)
        jac = f_y(y[m], h, N, lamb)
        f_vec = f(y[m], h, N, lamb)
        w_1 = linalg.solve(D_mat - alpha*tau*jac, f_vec)
        y[m+1] = y[m] + tau*w_1.real
        u[m+1, 0] = 0
        u[m+1, 1] = 1/4 * y[m+1, 0]
        u[m+1, 2:N+1] = y[m+1, :N-1]

    center_index = np.argmin(np.abs(np.linspace(a, b, N+1) - 0.5))
    return u[3, center_index]  # выбираем точку в которой будет происходить сгущение


# Метод Ричардсона
r = 2      # шаг сгущения
S = 6      # уровни сгущения
p = 2      # порядок схемы по пространству (примерно 2)
q = 1      # шаг сгущения

U = np.zeros((S, S))
R = np.zeros((S, S))

for s in range(S):
    N_s = r**s * 10
    U[s, 0] = solve(N_s)

for s in range(1, S):
    for l in range(s):
        R[s, l] = (U[s, l] - U[s-1, l]) / (r**(p + l*q) - 1)
        U[s, l+1] = U[s, l] + R[s, l]

# Печать таблиц
def PrintTriangular(A, i, title):
    print(title)
    print(' ', end=' ')
    for l in range(len(A)):
        print(f'p={p + l*q:<4d}', end=' ')
    print()
    for m in range(len(A)):
        print(f's={m:<2d}', end=' ')
        for l in range(m + 1 - i):
            print(f'{A[m, l]:7.6f}', end=' ')
        print()
    print()


# Таблица эффективного порядка точности
P = np.zeros((S, S))

for s in range(2, S):
    for l in range(s-1):
        num = np.abs(U[s-1, l] - U[s-2, l])
        den = np.abs(U[s, l] - U[s-1, l])
        P[s, l] = np.log(num / den) / np.log(r)

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
                print(f"{'—':>7}", end=' ')
            else:
                print(f'{A[m, l]:7.4f}', end=' ')
        print()
    print()

PrintTriangular(U, 0, 'Таблица приближённых значений решения:')
PrintTriangular(R, 1, 'Таблица оценок ошибок:')
PrintTriangularP(P, 2, 'Таблица эффективного порядка точности:')
