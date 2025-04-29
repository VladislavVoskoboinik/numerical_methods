import numpy as np
from matplotlib.pyplot import figure, xlabel, ylabel, title, show
from mpl_toolkits.mplot3d import Axes3D


N = 10
M = 500
h = 1/(N-1)
tau = 1/(M-1)
e = np.e


x = np.linspace(0, 1, N)
t = np.linspace(0, 1, M)
tgrid, xgrid = np.meshgrid(t, x)


y_analytic = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        y_analytic[i,j] = 1 + (x[i]**2) * np.exp(t[j]) / 2


y_explicit = np.zeros((N, M))
y_explicit[:, 0] = 1 + x**2/2

for j in range(M-1):
    if j > 0:
        y_explicit[0][j] = y_explicit[1][j]
        y_explicit[-1][j] = y_explicit[-2][j] + np.exp(t[j])*h
    
    for i in range(1, N-1):
        y_explicit[i][j+1] = ((y_explicit[i-1][j] - 2*y_explicit[i][j] + y_explicit[i+1][j])/h**2 * tau 
                             + y_explicit[i][j] 
                             + np.exp(t[j])*(x[i]**2/2 - 1)*tau)
    

    y_explicit[-1, -1] = y_explicit[-2, -1] + np.exp(t[-2])*h
    y_explicit[0, -1] = y_explicit[1, -1]

y_symmetric = np.zeros((N, M))
y_symmetric[:, 0] = 1 + x**2/2

alpha = np.zeros(N-1)
beta = np.zeros(N-1)

A = 1/(2*h**2)
B = A
C = 2*A + 1/tau

for j in range(M-1):
    alpha[0] = 1/(1 + h**2/tau)
    beta[0] = alpha[0]*(y_symmetric[1, j] - y_symmetric[0, j] 
               + h**2/tau*y_symmetric[0, j] 
               - h**2*np.exp(t[j] + 0.5*tau))
    
    for i in range(1, N-1):
        F = (y_symmetric[i, j]/tau 
             + (y_symmetric[i-1, j] - 2*y_symmetric[i, j] + y_symmetric[i+1, j])/(2*h**2) 
             + np.exp(t[j] + 0.5*tau)*(x[i]**2/2 - 1))
        alpha[i] = B/(C - A*alpha[i-1])
        beta[i] = (A*beta[i-1] + F)/(C - A*alpha[i-1])
    
    kappa_2 = 1/(1 + h**2/tau)
    mu_2 = kappa_2*(-(y_symmetric[-1, j] - y_symmetric[-2, j]) 
            + 2*h*np.exp(t[j] + 0.5*tau) 
            + h**2/tau*y_symmetric[-1, j] 
            - h**2/2*np.exp(t[j] + 0.5*tau))
    
    y_symmetric[-1, j+1] = (mu_2 + kappa_2*beta[-2])/(1 - kappa_2*alpha[-2])
    
    for i in reversed(range(N-1)):
        y_symmetric[i, j+1] = alpha[i]*y_symmetric[i+1, j+1] + beta[i]


y_implicit = np.zeros((N, M))
y_implicit[:, 0] = 1 + x**2/2

alpha = np.zeros(N-1)
beta = np.zeros(N-1)

A = 1/h**2
B = A
C = 2*A + 1/tau

for j in range(M-1):
    alpha[0] = 1/(1 + h**2/(2*tau))
    beta[0] = alpha[0]*(h**2*y_implicit[0, j]/(2*tau) 
              - h**2*np.exp(t[j+1]))
    
    for i in range(1, N-1):
        F = y_implicit[i, j]/tau + np.exp(t[j+1])*(x[i]**2/2 - 1)
        alpha[i] = B/(C - A*alpha[i-1])
        beta[i] = (A*beta[i-1] + F)/(C - A*alpha[i-1])
    
    kappa_2 = 1/(1 + h**2/(2*tau))
    mu_2 = kappa_2*(h**2/(2*tau)*y_implicit[-1, j] 
           - h**2/4*np.exp(t[j+1]) 
           + h*np.exp(t[j+1]))
    
    y_implicit[-1, j+1] = (mu_2 + kappa_2*beta[-2])/(1 - kappa_2*alpha[-2])
    
    for i in reversed(range(N-1)):
        y_implicit[i, j+1] = alpha[i]*y_implicit[i+1, j+1] + beta[i]



def create_figure(data, error, title_main, title_error, color, cmap='coolwarm'):
    # Основной график
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(tgrid, xgrid, data, cmap=cmap, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Время (t)')
    ax.set_ylabel('Пространство (x)')
    ax.set_zlabel('Температура')
    ax.set_title(title_main)
    
    # График ошибки
    fig_err = figure()
    ax_err = fig_err.add_subplot(111, projection='3d')
    surf_err = ax_err.plot_surface(tgrid, xgrid, error, cmap='viridis', antialiased=True)
    fig_err.colorbar(surf_err, shrink=0.5, aspect=5)
    ax_err.set_xlabel('Время (t)')
    ax_err.set_ylabel('Пространство (x)')
    ax_err.set_zlabel('Ошибка')
    ax_err.set_title(title_error)

# Аналитическое решение с новой цветовой картой
fig = figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(tgrid, xgrid, y_analytic, cmap='plasma', antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Время (t)')
ax.set_ylabel('Пространство (x)')
ax.set_zlabel('Температура')
ax.set_title('Аналитическое решение')

# Графики для всех схем с разными цветовыми картами
schemes = [
    (y_explicit, y_explicit - y_analytic, 
     'Явная схема', 'Ошибка явной схемы', 'cool'),
    (y_symmetric, y_symmetric - y_analytic, 
     'Симметричная схема', 'Ошибка симметричной схемы', 'twilight'),
    (y_implicit, y_implicit - y_analytic, 
     'Неявная схема', 'Ошибка неявной схемы', 'spring')
]

for data, error, name, err_name, cmap in schemes:
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(tgrid, xgrid, data, cmap=cmap, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Время (t)')
    ax.set_ylabel('Пространство (x)')
    ax.set_zlabel('Температура')
    ax.set_title(name)
    
    fig_err = figure()
    ax_err = fig_err.add_subplot(111, projection='3d')
    surf_err = ax_err.plot_surface(tgrid, xgrid, error, cmap='viridis', antialiased=True)
    fig_err.colorbar(surf_err, shrink=0.5, aspect=5)
    ax_err.set_xlabel('Время (t)')
    ax_err.set_ylabel('Пространство (x)')
    ax_err.set_zlabel('Ошибка')
    ax_err.set_title(err_name)

show()