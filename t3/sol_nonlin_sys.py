#! /bin/python3.10

# Номер IV.12.5 (г)

import math
import matplotlib.pyplot as plt
import numpy as np

def J(x: float, y:float):
    ret = np.zeros((2, 2))
    ret[0][0] = math.cos(x + 2)
    ret[0][1] = -1
    ret[1][0] = 1
    ret[1][1] = -math.sin(y - 2)
    return ret

"""
The chosen system is as follows:
            { sin(x + 1) - y = 1.2
            { 2x + cos(y) = 2
"""
def F1(x):
    F = np.array([[0.0], [0.0]])
    F[0] = np.sin(x[0] + 1.0) - x[1] - 1.2
    F[1] = 2 * x[0] + np.cos(x[1]) - 2.0

    return F

def F1_x(x: float, y: float) -> float:
    return 1.0 - 0.5 * math.cos(y)

def F1_y(x: float, y: float) -> float:
    return math.sin(x + 1) - 1.2

def J1(x):
    J = np.identity(2)
    J[0, 0] = np.cos(x[0] + 1.0)
    J[0, 1] = -1.0
    J[1, 0] = 2.0
    J[1, 1] = -np.sin(x[1])

    return J

def norm3(V):
    return float(np.sqrt(np.dot(V.T, V)))

def NewtonMethod(start, F, J, epsilon, norm):
    x = start
    xarr = [x[0]]
    yarr = [x[1]]
    while norm(F(x)) > epsilon:
        x = x - np.matmul(np.linalg.inv(J(x)), F(x))
        xarr.append(x[0])
        yarr.append(x[1])
        
    plt.grid()
    plt.title("Newton method")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(xarr, yarr, '.-', ms=5.0)
    plt.savefig("img/system_newton.png")
    plt.clf()

    return x

def MPI(start, F_x, F_y, epsilon):
    xP = 0
    yP = 0
    xarr = [xP]
    yarr = [yP]
    xN = F_x(xP, yP)
    yN = F_y(xP, yP)

    while (abs(xN - xP) > epsilon) and (abs(yN - yP) > epsilon):
        xP = xN
        yP = yN
        xarr.append(xP)
        yarr.append(yP)
        xN = F_x(xP, yP)
        yN = F_y(xP, yP)

    plt.grid()
    plt.title("MPI method")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(xarr, yarr, '.-', ms=8.0)
    plt.savefig("img/system_mpi.png")
    plt.clf()

    return np.array([[xN], [yN]])

def main():
    # MPI
    print("1. MPI:")
    start = np.array([[10.0], [10.0]])
    sol = MPI(start, F1_x, F1_y, 1e-6)
    print(f"Solution: {sol}")

    # Newton
    print("2. Newton method:")
    start = np.array([[0], [0]])
    sol = NewtonMethod(start, F1, J1, 1e-6, norm3)
    print(f"Solution: {sol}")

if __name__ == '__main__':
    main()
