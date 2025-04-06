import numpy as np
import matplotlib.pyplot as plt
import math


x_0 = np.pi/2# Начальное значение x
y_0 = 1         # Начальное значение y
v_0 = 0         # Начальное значение v (y')
w_0 = 1     # Начальное значение w (y'')
X = 10    #Конец отрезка
M = 100
tau = 0.2
p = 3; q = 3; S = 3
r = 2
eps = 1e-6
eps_loc = 1e-12
u_0 = [y_0, v_0, w_0]


def f(u, x) -> np.array:
    f = np.empty(3)
    f[0] = u[1]
    f[1] = u[2]
    f[2] = 2 * (u[2] - 1) * np.cos(x)/np.sin(x) 

    return f


def ERK3_step(current_x, current_u, current_step) -> np.array:
    w_1 = f(current_u, current_x)
    w_2 = f(current_u + 1/2*current_step*w_1, current_x + 1/2 * current_step)
    w_3 = f(current_u + 3/4 * current_step * w_2, current_x + 3/4 * current_step)
    return current_u + current_step * (2/9*w_1 + 3/9 * w_2 + 4/9 * w_3)


def ERK3(x):
    u = np.empty((len(x) + 1, 3))
    u[0] = [x_0, y_0, w_0]
    for m in range(len(x)-1):
        tau = x[m+1] - x[m]
        w_1 = f(u[m], x[m])
        w_2 = f(u[m] + 1/2*tau*w_1, x[m] + 1/2 * tau)
        w_3 = f(u[m] + 3/4 * tau * w_2, x[m] + 3/4 * x[m])
        u[m+1] = u[m] + tau * (2/9 * w_1 + 3/9 * w_2 + 4/9 * w_3)
    return u


def ERK3_local_thickening(M, eps, x_0, X) -> tuple[np.array, np.array]:
    tau = (X - x_0) / M
    u = [[y_0, v_0, w_0]]
    u_thickened = [[y_0, v_0, w_0]]
    x = [x_0]
    m = 0
    while x[-1] < X and m <= M:
        u.append(ERK3_step(x[-1], u[-1], tau))
        u_thickened.append(ERK3_step(x[-1], u_thickened[-1], tau/r))
        u_thickened.append(ERK3_step(x[-1] + tau/r, u_thickened[-1], tau/r))
        print(u_thickened[-1], u[-1])
        #error = np.sqrt(sum((u[m][j]**2 - u_thickened[m][j] ** 2) for i in range(3) for j in range(3)))
        error = np.linalg.norm(u[-1][0] - u_thickened[-1][0])

        x.append(x[-1] + tau)
        if error > eps_loc:  # Защита от слишком малых значений
            tau = ((eps_loc * (r**p - 1) * tau**(p+1)) / (error*(X-x_0))) **(1/p)
            #tau = min(tau_new, 1.1*tau) 
        else:
            tau = 1.1*tau  # Если ошибка слишком мала, увеличиваем шаг

        print(f'x[-1]={x[-1]} : tau={tau}')
        
        m += 1

    x = np.array(x)
    u = np.array(u)

    return np.array(x), np.array(u)


#U = np.zeros((S,S))
R = np.zeros((S,S))
p_eff = np.zeros((S,S))
x, u = ERK3_local_thickening(M, eps_loc, x_0, X)
U = np.zeros((S,S, len(x)+1))
U[0, 0] = u[:, 0]
s = 1
y_cur = U[0, 0, :]
y_last = np.empty(len(x))
while np.linalg.norm(y_cur - y_last) > eps and s < S:
        u = ERK3(x)
        print(U[s, 0].size)      
        print(u[:, 0].size)
        U[s,0] = u[:, 0]
        y_last = y_cur
        y_cur = U[s, 0]
        s += 1


for s in range(1,S):
    for l in range(s):
        R[s,l] = (np.linalg.norm(U[s,l] - U[s-1,l]))/(r**(p + l*q) - 1)
        U[s,l+1] = U[s,l] + R[s,l]

for s in range(2,S) :
    for l in range(s-1) :
        p_eff[s,l] = np.log(abs(R[s-1,l]/R[s,l]))/np.log(r)

# Функция выводит форматированную таблицу
def PrintTriangular(A,i) :
    print(' ',end=' ')
    for l in range(len(A)) :
        print(' p={0:<4d}'.format(p + l*q),end=' ')
    print()
    for m in range(len(A)) :
        print('s={0:<2d}'.format(m),end=' ')
        for l in range(m + 1 - i) :
            print('{0:7.4f}'.format(A[m,l]),end=' ')
        print()
    print()

print('Таблица приближённых значений интеграла:')
PrintTriangular(U,0)
print('Таблица оценок ошибок:')
PrintTriangular(R,1)
print('Таблица эффективных порядков точности:')
PrintTriangular(p_eff,2)

plt.plot([r**s*M for s in range(1,S)],abs(R[1:,0]),'-bo')
plt.xscale('log'); plt.yscale('log')