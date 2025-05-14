import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#get_ipython().run_line_magic('matplotlib', 'qt')

#   Неявная схема для уравнения теплопроводности на отрезке

# Создание сетки:----------------------------------------------------------
N=500 # число интервалов по x
M=2*N^2 # число интервалов по t должно удовлетворять условию
                      # M>=2*N^2
case_bound=1 # выбор варианта граничного условия
                      
x=np.zeros(N+1)
h=2/N
i=0
while i<=N:
    x[i]=i*h
    i+=1

t=np.zeros(M+1)
tau=1/M
j=0
while j<=M:
    t[j]=j*tau
    j+=1
#--------------------------------------------------------------------------

# численное решение:-------------------------------------------------------
y=np.zeros((M+1,N+1))

# начальное условие:
i=0
while i<=N:
    y[0][i]=3 - x[i] + np.cos(3*x[i]*np.pi/4)
    i+=1
    
# Создание массивов прогоночных коэффициентов:
alpha=np.zeros(N)
beta=np.zeros(N)
# Параметры системы:
A=tau/(2*h**2)
B=A
C=2*A+1

j=0
while j<=M-1:
    # прямой ход прогонки:
    i=1
    alpha[0] = 1
    beta[0] = h
    while i<=N-1:
        F=y[j][i]+tau*(y[j][i-1]-2*y[j][i]+y[j][i+1])/(2*h**2)
        alpha[i]=B/(C-A*alpha[i-1])
        beta[i]=(A*beta[i-1]+F)/(C-A*alpha[i-1])
        i+=1
    
    # первый вариант граничного условия при x=1:

    kappa_2=0
    mu_2=1    
    # обратный ход прогонки:
    y[j+1][N]=(mu_2 + kappa_2*beta[N-1])/(1-kappa_2*alpha[N-1])
    i=N-1
    while i>=0:
        y[j+1][i]=alpha[i]*y[j+1][i+1]+beta[i]
        i=i-1
    
    j+=1
    

#--------------------------------------------------------------------------

# аналитическое решение и вычисление погрешности:--------------------------
u=np.zeros((M+1,N+1))
err=np.zeros((M+1,N+1))

def u_a(x_val, t_val):

    return 3 - x_val + np.cos(3 * np.pi * x_val / 4) * np.exp(-(3 * np.pi / 4)**2 * t_val)


i=0
while i<=N:
    j=0
    while j<=M:
        u[j][i]=u_a(x[i],t[j])
        err[j][i]=u[j][i]-y[j][i]
        j+=1
    i+=1


# норма погрешности на последнем слое:
max_err=abs(err[M][0])
i=1
while i<=N:
    if max_err<=abs(err[M][i]):
        max_err=abs(err[M][i])
    i+=1
print(max_err)
#--------------------------------------------------------------------------

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
ax3.set_zlabel('y')
ax3.set_title('Погрешность')
plt.show()
