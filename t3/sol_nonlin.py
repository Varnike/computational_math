import math
import matplotlib.pyplot as plt
import numpy as np

def deriv(func, x: float, h: float) -> float:
    return (func(x + h) - func(x)) / h

def func(x) -> float:
    return (x ** 2) - np.exp(x) / 5

# For [-1, 0]
def iter_method(x) -> float:
    return -np.sqrt(np.exp(x) / 5)

def NewtonMethod(start, h, F, epsilon):
    x = start
    
    der = deriv(F, start, h)
    x = start - F(x)/der
    arr = [x]

    while np.abs(F(x)) > epsilon:
        der = deriv(F, x, h)
        x -= F(x)/der
        arr.append(x)

    plt.grid()

    plt.title("Newton method")
    plt.ylabel("X")
    plt.xlabel("Step")

    plt.plot(arr, '.-', ms=10.0)
    plt.savefig("img/eq_newton.png")
    plt.clf()

    return x

def FixedPointIteration(start, method, epsilon, f):
    x = start

    arr = [start]
    iters = 0
    while np.abs(f(x)) > epsilon:
        x = method(x)
        arr.append(x)

    plt.grid()

    plt.title("MPI method")
    plt.ylabel("X")
    plt.xlabel("Step")

    plt.plot(arr, '.-', ms=5.0)
    plt.savefig("img/eq_mpi.png")
    plt.clf()

    return x




def main():
    # MPI
    print("1. MPI method for x^2 - e^x / 5 = 0:")
    sol = FixedPointIteration(-1, iter_method, 1e-6, func)
    print(f"Solution: {sol}")

    # Newton
    print("2. Newton method:")
    sol = NewtonMethod(-1, 1e-3, func, 10e-6)
    print(f"Solution: {sol}")
if __name__ == '__main__':
    main()
