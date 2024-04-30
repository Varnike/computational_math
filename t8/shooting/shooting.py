import numpy as np
import matplotlib.pyplot as plt

# Метод стрельбы для задачи XI.9.3(a)
def d2y(x, y):
    return x * np.sqrt(y)

def get_Y(X, y0, alpha, h):
    Y = []
    y = y0
    dy = alpha
    for x in X:
        Y.append(y)
        y += dy * h
        dy = f(x, dy, d2y)
    plt.plot(X, Y)
    return Y

def f(x, f0, df):
    h = 4e-5
    fn = f0 + df(x, f0) * h
    for i in range(1000):
        if abs(fn - f0) <= 1e-4:
            return fn
        f0 = fn
        fn = f0 + df(x, f0) * h

    return fn

def shooting(X, y0, y1, alpha, ha, h, eps):
    Y = get_Y(X, y0, alpha, h)
    F = Y[-1] - y1

    while abs(F) > eps:
        Y = get_Y(X, y0, alpha + ha, h)
        alpha = alpha - F/((Y[-1] - y1 - F) / ha)
        Y = get_Y(X, y0, alpha, h)
        F = Y[-1] - y1

    return Y, alpha

def plot_solution(X, Y, alpha):
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X, Y, label="solution")
    plt.title(f"alpha = {alpha:.5f}")
    plt.show()

def main():
    y0 = 0
    y1 = 2
    alpha = 0
    eps = 1e-7
    h = 4e-5
    X = np.arange(0.0, 1.0, h)
    Y, alpha = shooting(X, y0, y1, alpha, 1e-2, h, eps)
    plot_solution(X, Y, alpha)

if __name__ == "__main__":
  main()
