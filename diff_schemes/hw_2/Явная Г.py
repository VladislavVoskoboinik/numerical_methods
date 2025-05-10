import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

#   Явная схема для уравнения теплопроводности на отрезке


#   ВАРИАНТ Г


# Создание сетки:----------------------------------------------------------
N=10 # число интервалов по x
M=10+2*(N**2) # число интервалов по t должно удовлетворять условию
                      # M>=2*N^2
                       
x=np.zeros(N+1)
L = 2
h=L/N
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
case_bound=1 # выбор варианта граничного условия
# начальное условие:
i=0
while i<=N:
    y[0][i]=3 - x[i] + np.cos(3*x[i]*np.pi/4)
    i+=1

j=0
while j<=M-1:
    # решение уравнения на внутренних точках:
    i=1
    while i<=N-1:
        y[j+1][i]=y[j][i]+tau*(y[j][i+1]-2*y[j][i]+y[j][i-1])/(h*h)
        i+=1
    # граничные условия:
    y[j+1][N]=1 # в данном случае не обязательно, так как y излачально 
                # заполнено нулями
    # первый вариант граничного условия при x=1:  
    if case_bound==1:
        y[j+1][0]=y[j+1][1]+h
        j+=1
'''    # второй вариант граничного условия при x=1: 
    if case_bound==2:
        y[j+1][N]=(y[j+1][N-1]+h*t[j+1]+h**2*y[j][N]/(2*tau)+h**2/2)/(1+h**2/(2*tau))
    # третий вариант граничного условия при x=1: 
    if case_bound==3:
        y[j+1][N]=4*y[j+1][N-1]/3-y[j+1][N-2]/3+2*h*t[j+1]/3
        '''

    
#--------------------------------------------------------------------------

# аналитическое решение и вычисление погрешности:--------------------------
u=np.zeros((M+1,N+1))
err=np.zeros((M+1,N+1))

def u_a(x,t,N):
    y_a = 0
    for n in range(1,N+1):
        y_a = -1/2*(4/(2*n-3)/np.pi*(np.cos((2*n-3)/2*np.pi) - 1) + 4/(2*n+3)/np.pi*(np.cos((2*n+3)/2*np.pi) - 1)) * np.exp(-(np.pi*n/2)**2*t)*np.sin(np.pi*n/2*x) + y_a
    y_a = y_a - x + 3
    return y_a

N_s = 25#Число членов ряда решения

i=0
while i<=N:
    j=0
    while j<=M:
        u[j][i]=u_a(x[i],t[j],N_s)
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

X, T = np.meshgrid(x, t)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, T, y, cmap='inferno')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('y')
ax2.set_title('Численное решение')
#plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, T, err, cmap='inferno')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('y')
ax3.set_title('Погрешность')
plt.show()
