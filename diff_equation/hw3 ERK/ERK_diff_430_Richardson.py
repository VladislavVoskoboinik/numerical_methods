import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
x_0 = np.pi/2
y_0 = 1
v_0 = 0
w_0 = 1
X = 10
M_initial = 100  # Начальное число шагов
r = 2           # Коэффициент сгущения
p = 3           # Порядок метода
S = 5     # Число уровней сгущения
eps = 1e-6      # Требуемая точность

u_0 = np.array([y_0, v_0, w_0])

def f(u, x):
    cot_x = np.cos(x)/(np.sin(x) + 1e-12)
    return np.array([u[1], u[2], 2*(u[2]-1)*cot_x])

def ERK3_step(current_x, current_u, tau):
    k1 = f(current_u, current_x)
    k2 = f(current_u + 0.5*tau*k1, current_x + 0.5*tau)
    k3 = f(current_u + 0.75*tau*k2, current_x + 0.75*tau)
    return current_u + tau*(2/9*k1 + 1/3*k2 + 4/9*k3)

def ERK3(M, u_initial, x_start, x_end):
    """Возвращает сетку и решение на всей сетке"""
    x = np.linspace(x_start, x_end, M+1)
    tau = (x_end - x_start)/M
    u = np.zeros((M+1, 3))
    u[0] = u_initial
    for m in range(M):
        u[m+1] = ERK3_step(x[m], u[m], tau)
    return x, u


def ERK3_on_grid(M, u_initial, x):
    """Возвращает сетку и решение на всей сетке"""
    u = np.zeros((M+1, 3))
    u[0] = u_initial
    for m in range(M):
        u[m+1] = ERK3_step(x[m], u[m], x[m+1] - x[m])
    return x, u 

def restrict_solution(fine_x, fine_u, coarse_x):
    """Интерполяция решения с мелкой сетки на грубую"""
    coarse_u = np.zeros((len(coarse_x), 3))
    for i in range(3):
        coarse_u[:,i] = np.interp(coarse_x, fine_x, fine_u[:,i])
    return coarse_u

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
        if error > eps:  # Защита от слишком малых значений
            tau = ((eps * (r**p - 1) * tau**(p+1)) / (error*(X-x_0))) **(1/p)
            #tau = min(tau_new, 1.1*tau) 
        else:
            tau = 1.1*tau  # Если ошибка слишком мала, увеличиваем шаг

        print(f'x[-1]={x[-1]} : tau={tau}')
        
        m += 1

    x = np.array(x)
    u = np.array(u)

    return np.array(x), np.array(u)

# Генерация решений с разным числом шагов
x, u = ERK3_local_thickening(M_initial, eps, x_0, X)
solutions = [u]
grids = [x]
for s in range(1, S):
    M = M_initial * (r ** s)
    x, u = ERK3_on_grid(M, u_0, x)
    grids.append(x)
    solutions.append(u)

# Создание общей базовой сетки (самая грубая)
base_x = grids[0]

# Таблицы Ричардсона для всех узлов
U = np.zeros((S, S, len(base_x), 3))  # Экстраполированные решения
R = np.zeros((S, S, len(base_x), 3))  # Разности
p_eff = np.zeros((S, S, len(base_x), 3))  # Добавляем компоненты

# Инициализация нулевого уровня
U[0,0] = restrict_solution(grids[0], solutions[0], base_x)

# Заполнение таблиц
for s in range(1, S):
    # Интерполяция текущего решения на базовую сетку
    U[s,0] = restrict_solution(grids[s], solutions[s], base_x)
    
    # Экстраполяция Ричардсона
    for l in range(1, s+1):
        factor = r**(p + (l-1)*1)
        U[s,l] = (factor * U[s,l-1] - U[s-1,l-1]) / (factor - 1)
        R[s,l-1] = U[s,l-1] - U[s-1,l-1]



# Изменяем размерность массива для эффективных порядков
p_eff = np.zeros((S, S, len(base_x), 3))

for s in range(2, S):
    for l in range(1, s):
        with np.errstate(divide='ignore', invalid='ignore'):
            for comp in range(3):
                ratio = np.abs(R[s-1,l-1,:,comp] / R[s,l-1,:,comp])
                p_eff[s,l,:,comp] = np.log(ratio) / np.log(r)

# Функция для вывода таблицы средних порядков по сетке
def print_effective_orders(p_eff, component=0):
    """Выводит таблицу эффективных порядков для выбранной компоненты (0=y, 1=v, 2=w)"""
    print(f"\nТаблица эффективных порядков (компонента {'yvw'[component]}):")
    print("s\\l |", end="")
    for l in range(p_eff.shape[1]):
        print(f"  l={l:<7} |", end="")
    print("\n" + "-"*(12*p_eff.shape[1] + 1))
    
    for s in range(p_eff.shape[0]):
        print(f"{s:3} |", end="")
        for l in range(p_eff.shape[1]):
            if l >= s: 
                print(" "*(10) + " |", end="")
                continue
                
            # Усредняем по всем точкам, игнорируя NaN и inf
            with np.errstate(invalid='ignore'):
                mean_p = np.nanmean(p_eff[s,l,:,component])
                mean_p = np.where(np.isinf(mean_p), np.nan, mean_p)
                mean_p = np.nanmean(mean_p)
                
            print(f"  {mean_p:6.3f}   |", end="")
        print()

# Выводим таблицы для всех компонент
for comp in range(3):
    print_effective_orders(p_eff, component=comp)



def print_refined_values(U, R, component=0):
    """Выводит таблицу значений и уточнений для выбранной компоненты (0=y, 1=v, 2=w)"""
    print(f"\nТаблица значений (компонента {'yvw'[component]}):")
    print("s\\l |", end="")
    for l in range(U.shape[1]):
        print(f"      l={l:<12} |", end="")
    print("\n" + "-"*(20*U.shape[1] + 1))
    
    for s in range(U.shape[0]):
        print(f"{s:3} |", end="")
        for l in range(U.shape[1]):
            if l > s: 
                print(" "*(18) + " |", end="")
                continue
                
            # Берем последнюю точку (X)
            value = U[s,l,-1,component] if l == 0 else U[s,l,-1,component]
            error = R[s,l-1,-1,component] if l > 0 else 0
                
            if l == 0:
                print(f"  {value:12.6e}    |", end="")
            else:
                print(f"  {value:12.6e} ({error:+.2e}) |", end="")
        print()

# Выводим таблицы для всех компонент
for comp in range(3):
    print_refined_values(U, R, component=comp)
    print_effective_orders(p_eff, component=comp)  # Из предыдущего ответа


plt.figure()
for i in range(len(base_x)):
    plt.semilogy(base_x, p_eff[2,1,:,0], 'b-', alpha=0.3)  # comp=0 (y)
plt.xlabel('x')
plt.ylabel('Эффективный порядок')
plt.title('Распределение порядка точности по сетке (y компонента)')
plt.grid(True)
plt.show()