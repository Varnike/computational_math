import numpy as np
import matplotlib.pyplot as plt

# Метод прогонки для задачи XI.9.5

h = 0.005
N = int(1.0/h)

def f(x):
    return np.cos(2 * np.pi * x)

def P2(x):
    return 10 + np.sin(2 * np.pi * x)

def progon(A, B, C, r, N):
    a_arr = np.zeros(N)
    b_arr = np.zeros(N)
    g_arr = np.zeros(N)
    m_arr = np.zeros(N)
    n_arr = np.zeros(N)
    Y     = np.zeros(N)

    a_arr[1] =  C[0] / B[0]
    b_arr[1]  = -r[0] / B[0]
    g_arr[1] =  A[0] / B[0]

    for i in range(1, N - 1):
        a_arr[i + 1] = C[i] / (B[i] - a_arr[i] * A[i])
        b_arr[i + 1]  = (A[i] * b_arr[i] - r[i]) / (B[i] - a_arr[i] * A[i])
        g_arr[i + 1] = (A[i] * g_arr[i]) / (B[i] - a_arr[i] * A[i])

    m_arr[N-1] = -C[N-1] / (A[N-1] * (a_arr[N-1] + g_arr[N-1]) - B[N-1])
    n_arr[N-1] = (r[N-1] - A[N-1] * b_arr[N - 1]) / (A[N-1] * (a_arr[N - 1] + g_arr[N - 1]) - B[N-1])

    for i in range(N - 2, -1, -1):
        m_arr[i] = a_arr[i + 1] * m_arr[i + 1] + g_arr[i + 1] * m_arr[N - 1]
        n_arr[i] = a_arr[i + 1] * n_arr[i + 1] + g_arr[i + 1] * n_arr[N - 1] + b_arr[i + 1]

    y0 = n_arr[0] / (1 - m_arr[0])
    yN = m_arr[N-1] * y0 + n_arr[N-1]
    Y[0] = y0
    Y[N-1] = yN


    for i in range(N - 1, 0, -1):
        Y[i - 1] = a_arr[i] * Y[i] + b_arr[i] + g_arr[i] * yN

    return Y

def plot_solutions(X, Y, h):
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Final solution, one period")
    plt.plot(X, Y)
    plt.show()

    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Final solution, 4 periods")
    plt.plot(np.arange(0, 4.0, h), np.tile(Y, 4))
    plt.show()

# initial values
A = np.zeros(N)
B = np.zeros(N)
C = np.zeros(N)
r = np.zeros(N)
h2 = h**2

for i in range(0, N):
    A[i] = 1.0
    B[i] = 2.0 + P2(i * h) * h2
    C[i] = 1.0
    r[i] = f(i * h) * h2

# solve and plot
Y = progon(A, B, C, r, N)
X = np.arange(0, 1.0, h)
plot_solutions(X, Y, h)
