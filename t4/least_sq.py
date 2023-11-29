import matplotlib.pyplot as plt
import numpy as np
import bisect

population = {
    1910: 92228496, 1920: 106021537, 1930: 123202624, 1940: 132164569,
    1950: 151325798, 1960: 179323175, 1970: 203211926, 1980: 226545805,
    1990: 248709873, 2000: 281421906
}

# setup plot
plt.title("Least square approximation")
plt.ylabel("population")
plt.xlabel("year")

x = list(population.keys())
y = list(population.values())
ln = len(y)



p = []
for i in range(3):
    p.append(lambda x, i=i: x ** (2 - i))

matrix_size = 3
matrix = np.array([[0.0] * matrix_size for _ in range(matrix_size)])
print(matrix)

print(p)
for i in range(3):
    for j in range(3):
        for year in x:
            matrix[i][j] += p[i](year) * p[j](year)
r = [0.0] * 3
for i in range(3):
    for j in range(ln):
        r[i] += y[j] * (x[j] ** (2 - i))
coeffs = np.linalg.solve(matrix, r)

#--------------------------------------------------------------------------------------
x_plot = []
for i in range(1910, 2020, 10):
    x_plot.append(i)

app_plot = [0] * len(x_plot)
for k in range(0, len(x_plot)):
    app_plot[k] = coeffs[0] * x_plot[k] ** 2 + coeffs[1] * x_plot[k] + coeffs[2]

plt.plot(x_plot, app_plot)
plt.savefig("img/least_square.png")
plt.show()

