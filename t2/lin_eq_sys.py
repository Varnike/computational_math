import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import scipy
import scipy.linalg

import scipy
import matplotlib.pyplot as plt
import math
# Solve equation Ax = b
class LinEqSys:
    def __init__(self, matrix, f):
        assert np.shape(matrix)[0] == np.shape(matrix)[1]
        assert np.shape(matrix)[0] == np.shape(f)[0]

        self.matrix    = matrix
        self.f         = f

        self.dimension = np.shape(f)[0]
        self.solution  = np.zeros(self.dimension)
        self.is_solved = False

    def solve_eq(self, solution_method):
        self.solution = solution_method(self)
        self.is_solved = True

# Gauss method
def gauss(A, b):
    n = len(A)
    M = A

    M = np.hstack((M,np.array([b]).T))

    for i in range(n):

        leading = i + np.argmax(np.abs(A[:,i][i:]))
        M[[i, leading]] = M[[leading, i]] 

        M[i] /= M[i][i]
        row = M[i]

        for r in M[i + 1:]:
            r -= r[i] * row

    for i in range(n - 1, 0, -1):
        row = M[i]
        for r in reversed(M[:i]):
            r -= r[i] * row

    return M[:,-1]

# Get initial linear equation system
def get_eq_system(n):
    matrix = np.full((n, n), 0.0)

    for i in range(0, n):
        matrix[0, i] = 1

    for i in range(1, n - 1):
        matrix[i, i - 1] = 1
        matrix[i, i] = 10
        matrix[i, i + 1] = 1

    matrix[n - 1, n - 1] = 1
    matrix[n - 1, n - 2] = 1

    f = np.array([n - i for i in range(0, n)])

    return matrix, f
        
# LU decomposition method
def minors_degenerated(matrix):
    assert np.shape(matrix)[0] == np.shape(matrix)[1]
    n = np.shape(matrix)[0]

    for i in range(1, n + 1):
        if np.linalg.det(matrix[np.ix_([j for j in range(i)], [j for j in range(i)])]) == 0:
            return True

    return False

def LU_decomposition(A, b, n):
    P, L, U = scipy.linalg.lu(A)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= y[j] * L[i, j]

    x = np.zeros(n)
    for i in range(n):
        x[n - 1 - i] = y[n - 1 - i]
        for j in range(1, i + 1):
            x[n - 1 - i] -= x[n - 1 - i + j] * U[n - 1 - i, n - 1 - i + j]

        x[n - 1 - i] /= U[n - 1 - i, n - 1 - i]

    return x

# Jacobi method
def jacobi(A, b, it, x):
    if x is None:
        x = zeros(len(A[0]))

    D = diag(A)
    R = A - diagflat(D)
    errs = np.zeros(it)

    for i in range(it):
        x = (b - dot(R,x)) / D
        diff = b - np.matmul(A, x)
        errs[i] = np.linalg.norm(diff)  

    return x, errs

# Jacobi method
def seidel(A, b, it, x):
    if x is None:
        x = zeros(len(A[0]))

    D = diagflat(diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    inv = np.linalg.inv(L + D)
    R = -dot(inv, U)
    F = dot(inv, b)
    errs = np.zeros(it)

    for i in range(it):
        x = (F + dot(R,x))
        diff = b - np.matmul(A, x)
        errs[i] = np.linalg.norm(diff)

    return x, errs

# Upper relaxation method
def upper_relaxation(A, b, w, N):
    x = np.zeros(len(A[0]))
    data = np.zeros(N)
    u = np.triu(A)
    l = np.tril(A)
    L = A - u
    D = l + u - A
    U = A - l
    B = - np.matmul(np.linalg.inv(D + w * L), (w - 1) * D + w * U)
    F =   np.matmul(np.linalg.inv(D + w * L), b) * w

    for i in range(N):
        x = np.matmul(B, x) + F
        diff = b - np.matmul(A, x)
        data[i] = np.linalg.norm(diff)

    return x, data

# Function for representing upper relaxation method output data
def plot_upper_relaxation(names, data):
    plt.figure(figsize=(13, 7))
    plt.title("Upper Relaxation method")
    plt.xlabel("iteration")    
    plt.ylabel("lg(Acc)") # Accuracy
    plt.yscale("log")
    for i in range(len(data)):
        iterations = [i for i in range(len(data[i]))]
        plt.plot(iterations, data[i], ".-", label=names[i])
        plt.legend()

    plt.show()

def plot_error(plot_name, data):
    plt.figure(figsize=(13, 7))
    plt.xlabel("iteration")
    plt.ylabel("lg(Acc)") # Accuracy
    plt.yscale("log")
    plt.title(plot_name)
    it = [i for i in range(len(data))]
    plt.plot(it, data, ".-")

    plt.show()

#################################################

N = 100 # Number of linear equations
ITER = 25 # Number of iterations in iterations methods
EPS = 1e-6 # Precision of methods

# Execution start
A, b = get_eq_system(N)
print("Initial equation system:")
print(A)
print(b)

# Test all methods

print("1) Gaussian elimination method")
A, b = get_eq_system(N)
sol = gauss(A, b)
if np.linalg.norm(np.matmul(A,sol) - b) > EPS:
    print("Error: wrong solution")
else:
    print("OK")

print("2) LU-decomposition method")
A, b = get_eq_system(N)
if minors_degenerated(A) == False:
    sol = LU_decomposition(A, b, N)
    if np.linalg.norm(np.matmul(A, sol) - b) > EPS:
        print("Error: wrong solution")
    else:
        print("OK")

else:
    print("\nCan't decompose A to LU!")

# 3) Seidel method
A, b = get_eq_system(N)
x0 = np.zeros(N)
sol, errs = seidel(A, b, ITER, x0)
plot_error("Seidel method", errs)

# 4) Jacoby method
A, b = get_eq_system(N)
x0 = np.zeros(N)
sol, errs = jacobi(A, b, ITER, x0)
plot_error("Jacoby method", errs)

# 5) For upper relaxation method we need to 'guess' optimal w
data = []
names = []
for w in np.arange(1.0, 2.2, 0.1):
    w = math.ceil(w * 10) / 10
    x, dt = upper_relaxation(A, b, w, ITER)
    data.append(dt)
    names.append("w=" + str(w))

plot_upper_relaxation(names, data)

print(sol)
