import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import math

def method_progonka(N, a, b, c, f, parameter):
    Y = zeros(N + 1)
    alpha = zeros(N)
    beta = zeros(N)
    alpha[0] = parameter[2] / (-parameter[4])
    beta[0] = parameter[0] / parameter[4]

    for n in range(0, N - 1, 1):
        alpha[n + 1] = b / (-c - a * alpha[n])
        beta[n + 1] = (a * beta[n] - f[n + 1]) / (-c - a * alpha[n])

    Y[N] = (-parameter[1] + parameter[3] * beta[N - 1]) / (-parameter[5] - parameter[3] * alpha[N - 1])
    for n in range(N - 1, -1, -1):
        Y[n] = alpha[n] * Y[n + 1] + beta[n]
    return Y

a = 1; b = 1; c = -2
N = 30; h = 1 / N
cond = 2  # Use cond=2 for Dirichlet boundary conditions
f = zeros(N + 1)

u = zeros(N + 1)
x = zeros(N + 1)
for n in range(0, N + 1, 1):
    x[n] = n * h
    f[n] = - (h ** 2 * math.exp(x[n]))  # Corrected sign for the source term
    u[n] = - (math.exp(x[n]) - x[n] + math.e - 1)  # Analytical solution

# Set boundary conditions parameters for Dirichlet
if cond == 2:
    u0_val = -math.e
    uN_val = -2 * (math.e - 1)
    parameter = [u0_val, uN_val, 0, 0, 1, 1]  # Correct Dirichlet parameters

Y = method_progonka(N, a, b, c, f, parameter)

# Plotting
fig, ax = plt.subplots()
plt.plot(x, u, label='Exact solution')
plt.scatter(x, Y, color='red', label='Approximate solution', marker='x')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Exact and approximate solution (Dirichlet)')
plt.legend()
plt.show()

# Error analysis
NT = 10
N_values = np.zeros(NT)
h_values = np.zeros(NT)
R = np.zeros(NT)
N_values[0] = 10
for i in range(1, NT):
    N_values[i] = int(2 * N_values[i - 1])

for i in range(0, NT):
    h_values[i] = 1 / N_values[i]
    current_N = int(N_values[i])
    f = zeros(current_N + 1)
    u = zeros(current_N + 1)
    x = zeros(current_N + 1)
    for n in range(0, current_N + 1):
        x[n] = n * h_values[i]
        f[n] = - (h_values[i] ** 2 * math.exp(x[n]))  # Corrected sign
        u[n] = - (math.exp(x[n]) - x[n] + math.e - 1)
    
    # Dirichlet parameters
    parameter = [-math.e, -2 * (math.e - 1), 0, 0, 1, 1]
    Y = method_progonka(current_N, a, b, c, f, parameter)

    # L2 error calculation
    s = ((u[0] - Y[0]) ** 2) * h_values[i] / 2
    for n in range(1, current_N):
        s += ((u[n] - Y[n]) ** 2) * h_values[i]
    s += ((u[current_N] - Y[current_N]) ** 2) * h_values[i] / 2
    R[i] = math.sqrt(s)

# Plot error
fig, ax = plt.subplots()
plt.loglog(h_values, R, 'b-', label='Error norm')
plt.scatter(h_values, R, color='red')
ax.set_xlabel('Step h (log scale)')
ax.set_ylabel('Error norm (log scale)')
ax.set_title('Convergence Rate (Dirichlet)')
plt.grid(True, which='both')
plt.legend()
plt.show()