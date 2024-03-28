import statistics
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

# y'' + e * (y ^ 2 - 1)y' + y = 0
def func_lab(t, y):
    return np.array([y[1], math.exp(1) * (1 - y[0] ** 2) * y[1] - y[0]])

# Runge–Kutta 4th order method
def RK_4th_ord(h, t, y, func):
    stepcnt = np.shape(y)[0]
    for j in range(1, np.shape(t)[0]):
        k1 = func(t[j - 1], y[j - 1])
        k2 = func(t[j - 1] + 1 / 2 * h, y[j - 1] + 1 / 2 * h * k1)
        k3 = func(t[j - 1] + 1 / 2 * h, y[j - 1] + 1 / 2 * h * k2)
        k4 = func(t[j - 1] + 1 * h, y[j - 1] + 1 * h * k3)
        y[j] = y[j - 1] + h * (1 / 6 * k1 + 2 / 6 * k2 + 2 / 6 * k3 + 1 / 6 * k4)

    return t, y

# Adams 3rd order method
def adams_3d_ord(h, t, y, func):
    stepcnt = np.shape(y)[0]
    f = np.zeros((stepcnt, np.shape(y[0])[0]))
    f[0] = func(t[0], y[0])
    y[1] = y[0] + h * f[0]
    f[1] = func(t[1], y[1])
    y[2] = y[1] + h * f[1]
    f[2] = func(t[2], y[2])

    for j in range(3, np.shape(t)[0]):
        y[j] = y[j - 1] + \
            h * (23 / 12 * f[j - 1] - 16 / 12 * f[j - 2] + 5 / 12 * f[j - 2])
        f[j] = func(t[j], y[j])

    return t, y

# x_{n+1} = x_n + h * f(t_n, x_n)
def euler_1st_ord(h, t, y, func):
    stepcnt = np.shape(y)[0]
    for j in range(1, np.shape(t)[0]):
        k1_vec = func(t[j - 1], y[j - 1])
        y[j] = y[j - 1] + h * k1_vec

    return t, y

def eq_solver(h, t1, t2, x0, func, method):
    stepcnt = int((t2 - t1) / h) + 1
    t = np.linspace(t1, t2, stepcnt)
    out_vec = np.zeros((stepcnt, np.shape(x0)[0]))
    for i in range(np.shape(x0)[0]):
        out_vec[0][i] = x0[i]

    return method(h, t, out_vec, func)

def plot_phase(y1, y2, name, filename):
    plt.plot(y1, y2)
    plt.title(f"Phase diagram, {name}")
    plt.grid()
    plt.xlabel('y2')
    plt.ylabel('y1')
    plt.savefig(filename)
    plt.show()

def plot_yt(t, y, ind, name, filename):
    plt.plot(t, y)
    plt.title(f"y{ind}(t), {name}")
    plt.grid()
    plt.xlabel('y')
    plt.ylabel('t')
    plt.savefig(filename)
    plt.show()

tau = 0.07

t, sol_euler = eq_solver(tau, 0, 100, np.array([2, 0]), func_lab, euler_1st_ord)
y1 = np.array([sol_euler[i][0] for i in range(np.shape(sol_euler)[0])])
y2 = np.array([sol_euler[i][1] for i in range(np.shape(sol_euler)[0])])
plot_phase(y1, y2, "Euler 1st order", "pics/euler1_1.png")
plot_yt(t, y1, 0, "Euler 1st order", "pics/euler1_2.png")
plot_yt(t, y2, 1, "Euler 1st order", "pics/euler1_3.png")

t, sol_rk = eq_solver(tau, 0, 100, np.array([2, 0]), func_lab, RK_4th_ord)
y1 = np.array([sol_rk[i][0] for i in range(np.shape(sol_rk)[0])])
y2 = np.array([sol_rk[i][1] for i in range(np.shape(sol_rk)[0])])
plot_phase(y1, y2, "Runge–Kutta 4th order", "pics/rk4_1.png")
plot_yt(t, y1, 0, "Runge–Kutta 4th order", "pics/rk4_2.png")
plot_yt(t, y2, 1, "Runge–Kutta 4th order", "pics/rk4_3.png")

t, sol_adams = eq_solver(tau, 0, 100, np.array([2, 0]), func_lab, adams_3d_ord)
y1 = np.array([sol_adams[i][0] for i in range(np.shape(sol_adams)[0])])
y2 = np.array([sol_adams[i][1] for i in range(np.shape(sol_adams)[0])])
plot_phase(y1, y2, "Adams 3rd order", "pics/adams3_1.png")
plot_yt(t, y1, 0, "Adams 3rd order", "pics/adams3_2.png")
plot_yt(t, y2, 1, "Adams 3rd order", "pics/adams3_3.png")
