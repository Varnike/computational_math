import matplotlib.pyplot as plt
import numpy as np
import bisect

population = {
    1910: 92228496, 1920: 106021537, 1930: 123202624, 1940: 132164569,
    1950: 151325798, 1960: 179323175, 1970: 203211926, 1980: 226545805,
    1990: 248709873, 2000: 281421906
}
# setup plot
plt.title("Spline approximation")
plt.ylabel("population")
plt.xlabel("year")

x = list(population.keys())
y = list(population.values())
ln = len(y)

coeffs = [0] * ln
matrix = [[0] * (ln - 2) for _ in range(ln - 2)]

for i in range(ln - 2):
    for j in (-1, 0, 1):
        if not ((i + j) >= (ln - 2) or (i + j) < 0):
            matrix[i][i + j] = ((x[i + 1] - x[i]) * (j != 1) + (x[i + 2] - x[i + 1]) * (j != -1)) / (3 * (1 + (j != 0)))

coeffs_prev = np.linalg.solve(matrix, [((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i] / (x[i + 1] - x[i]))) for i in range(ln - 2)])
for i in range(ln - 2):
    coeffs[i + 1] = coeffs_prev[i]


#--------------------------------------------------------------------------------------
x_plot = []
for i in range(1910, 2020, 10):
    x_plot.append(i)

app_plot = [0] * len(x_plot)
for k in range(0, len(x_plot)):
    index = min(bisect.bisect_right(x, x_plot[k]), ln - 1)
    h = x[index] - x[index - 1]
    app_plot[k] = (x_plot[k] - x[index - 1]) ** 3 / (6 * h) * coeffs[index] + (x[index] - x_plot[k]) ** 3 / (6 * h) * coeffs[index - 1] + (x_plot[k] - x[index - 1]) / h * (y[index] - h**2 * coeffs[index] / 6) + (x[index] - x_plot[k]) / h * (y[index - 1] - h**2 * coeffs[index - 1] / 6)

plt.plot(x_plot, app_plot)
plt.savefig("images/spline.png")
plt.show()

