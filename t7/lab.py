import math
import numpy as np
from numpy import inf
import statistics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker   import MaxNLocator, ScalarFormatter, FixedFormatter

MAX_ITERATIONS = 30000

# Function from task
def func_lab(t, xv):
    return np.array([77.27 * (xv[1] + xv[0] * (1 - 8.375 * 10 ** (-6) * xv[0] - xv[1])),  1.0 / 77.27 * (xv[2]  - (1 + xv[0]) * xv[1]),  0.161 * (xv[0] - xv[2])])

# Its jacobian
def jacobian(xv):
    return np.array([[77.27 * (1 - xv[0] * 2 * 8.375 * 10 ** (-6) - xv[1]), 77.27 * (1 - xv[0]), 0], [-1/77.27 * xv[1], 1 / 77.27 * (-1 - xv[0]), 1 / 77.27], [0.161, 0, -0.161]])

def jacobian_gear(h, xv, x):
    return np.eye(np.shape(xv)[0]) - 6 * h / 11 * jacobian(x)

def gear_sys(h, function, xv, t, x_free_vec, x):
    return x - 6 * h / 11 * func_lab(t, x) + x_free_vec

def solve_eq(h, t_start, t_end, x0, method):
    stepcnt = int((t_end - t_start) / h) + 1
    t = np.linspace(t_start, t_end, stepcnt)
    out_vec = np.zeros((stepcnt, np.shape(x0)[0]))
    for i in range(np.shape(x0)[0]):
        out_vec[0][i] = x0[i]

    return method(h, t, out_vec)

def solve_sys(v0, eps, h, t, x_v1, x_v2):
        n_iters = 0
        kn = np.copy(v0)
        kp = np.copy(v0) + np.ones(len(v0)) * 2 * eps
        
        while np.abs(np.linalg.norm(kp, ord=inf) - np.linalg.norm(kn, ord=inf)) > eps and n_iters <= MAX_ITERATIONS:
            kp = np.copy(kn)
            kn = kp - np.dot(np.linalg.inv(jacobian_gear(h, x_v1, kp)), gear_sys(h, func_lab, x_v1, t, x_v2, kp))
            n_iters += 1

        return kn, n_iters

# Gears 3d order method
def gear_3d(h, t, xv):
    stepcnt = np.shape(xv)[0]
    f = np.zeros((stepcnt, np.shape(xv[0])[0]))
    f[0] = func_lab(t[0], xv[0])
    xv[1] = xv[0] + h * f[0]
    f[1] = func_lab(t[1], xv[1])
    xv[2] = xv[1] + h * f[1]
    f[2] = func_lab(t[2], xv[2])

    for j in range(3, np.shape(t)[0]):
        x1 = -18 / 11 * xv[j - 1] + 9 / 11 * xv[j - 2] - 2 / 11 * xv[j - 3]
        x0 = xv[j - 1] + h * (23 / 12 * f[j - 1] - 16 / 12 * f[j - 2] + 5 / 12 * f[j - 2])
        xv[j], n_iters = solve_sys(x0, 1e-6, h, t[j], xv[j], x1)

        f[j] = func_lab(t[j], xv[j])
    return t, xv


x0 = np.array([4, 1.1, 4])
tau = 0.007
t, x = solve_eq(tau, 0, 100, x0, gear_3d)

plt.figure(figsize=[15, 10])
colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(x[0])))
for ind in range(len(x[0])):
    plt.scatter(t[:len(x)], x[:,ind], color=colors[ind], label=f'x{ind+1}', s=0.1)

plt.title("Gear 3r order")
plt.grid()
plt.xlabel('t')
plt.ylabel('x')
plt.savefig("pics/Gear_3r_order.png")
plt.show()
