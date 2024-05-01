import numpy as np
import matplotlib.pyplot as plt

h   = 1e-4
eps = 1e-6

def y(Y, x, h):
    idx = int(x / h)
    return Y[idx]

def matrix_proc(X, Y, y0, y1, A, F, N, h):
    F[0]    = y0
    F[N]    = y1
    A[0][0] = 1.0    
    A[N][N] = 1.0
    for i in range(1, N):
        y_i         = y(Y, X[i], h)
        A[i][i - 1] = 1.0 / (h ** 2)
        A[i][i]     = -2.0 / (h ** 2) - X[i] / 2.0 / np.sqrt(y_i)
        A[i][i + 1] = 1.0 / (h ** 2) 
        F[i]        = X[i] / 2.0 * np.sqrt(y_i)

    return A, F

def plot_solution(X, Y):
    plt.plot(X, Y[:-1])
    plt.title(f"Solution")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def linear(X, Y, y0, y1, A, F, N, h, eps):
    it = 0
    A, F = matrix_proc(X, Y, y0, y1, A, F, N, h)
    Y_n  = np.linalg.solve(A, F)

    while np.max(np.abs(Y - Y_n)) > eps:
        print(f"iteration #{it}")
        it += 1

        Y    = Y_n
        A, F = matrix_proc(X, Y, y0, y1, A, F, N, h)
        Y_n  = np.linalg.solve(A, F)

    return X, Y

# initial values
y0 = 0
y1 = 2
X = np.arange(0.0, 1.0, h)
N = X.size
Y = np.full(N + 1, 1.0)
A = np.zeros((N + 1, N + 1))
F = np.zeros(N + 1)

# solve and plot solution
X, Y = linear(X, Y, y0, y1, A, F, N, h, eps)
plot_solution(X, Y)
