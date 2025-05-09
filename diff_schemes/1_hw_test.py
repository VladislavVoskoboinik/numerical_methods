import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import math

# Parameters: [0)f0, 1)fN, 2)b0, 3)aN, 4)c0, 5)cN]
# The system equations are:
# c0*u_0 + b0*u_1 = f0
# ai*u_{i-1} + ci*u_i + bi*u_{i+1} = fi for i=1,...,N-1
# aN*u_{N-1} + cN*u_N = fN
def method_progonka(N, a, b, c, f, parameter):
    Y = zeros(N + 1)
    alpha = zeros(N)
    beta = zeros(N)
    alpha[0] = parameter[2] / (-parameter[4])
    beta[0] = parameter[0] / parameter[4]

    # Forward sweep
    for n in range(0, N - 1, 1):
        alpha[n + 1] = b / (-c - a * alpha[n])
        beta[n + 1] = (a * beta[n] - f[n + 1]) / (-c - a * alpha[n])

    # Backward sweep
    Y[N] = (-parameter[1] + parameter[3] * beta[N - 1]) / (-parameter[5] - parameter[3] * alpha[N - 1])
    for n in range(N - 1, -1, -1):
        Y[n] = alpha[n] * Y[n + 1] + beta[n]
    return Y

a = 1; b = 1; c = -2
N = 500; h = 1 / N
cond = 1
f = zeros(N + 1)

# Initialize f with N+1 zeros, but f[0] and f[N] are set via parameters
# The actual f0 and fN are in the parameter array
u = zeros(N + 1)
x = zeros(N + 1)
for n in range(0, N + 1, 1):
    x[n] = n * h
    f[n] = h ** 2 * math.exp(x[n])
    u[n] = math.exp(x[n]) - x[n] + math.e - 1

# Set boundary conditions parameters
# parameter[0)f0, 1)fN, 2)b0, 3)aN, 4)c0, 5)cN
if cond == 1:
    parameter = [0, 0, -1, -1, 1, -h / 2 + 1]
if cond == 2:
    parameter = [-h ** 2 / 2, h ** 2 * math.e / 2, -1, 1, 1, h / 2 - 1]

Y = method_progonka(N, a, b, c, f, parameter)

# Plot exact and approximate solutions
fig, ax = plt.subplots()
plt.plot(x, u)
plt.scatter(x, Y, color='red')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Exact and approximate solution type {}'.format(cond))
plt.legend(['Exact solution', 'Approximate solution'])
plt.show()

# Error analysis for different step sizes
NT = 10  # Number of tests
N_values = zeros(NT)
h_values = zeros(NT)
R = zeros(NT)
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
        f[n] = h ** 2 * math.exp(x[n])
        u[n] = math.exp(x[n]) - x[n] + math.e - 1
    
    if cond == 1:
        parameter = [0, 0, -1, -1, 1, -h / 2 + 1]
    if cond == 2:
        parameter = [-h_values[i] ** 2 / 2, h_values[i] ** 2 * math.e / 2, -1, 1, 1, h_values[i] / 2 - 1]

    Y = method_progonka(current_N, a, b, c, f, parameter)

    # Calculate L2 error norm
    s = ((u[0] - Y[0]) ** 2) * h_values[i] / 2
    for n in range(1, current_N):
        s += ((u[n] - Y[n]) ** 2) * h_values[i]
    s += ((u[current_N] - Y[current_N]) ** 2) * h_values[i] / 2
    R[i] = math.sqrt(s)

# Plot error norm vs step size
fig, ax = plt.subplots()
plt.plot(h_values, R)
plt.scatter(h_values, R, color='red')
ax.set_xlabel('Step h')
ax.set_ylabel('Error norm')
ax.set_title('Error analysis type {}'.format(cond))
plt.show()