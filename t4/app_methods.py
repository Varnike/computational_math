import matplotlib.pyplot as plt
import numpy as np
import bisect

population = {
    1910: 92228496, 1920: 106021537, 1930: 123202624, 1940: 132164569,
    1950: 151325798, 1960: 179323175, 1970: 203211926, 1980: 226545805,
    1990: 248709873, 2000: 281421906
}

plt.title("Newton method")
plt.ylabel("population")
plt.xlabel("year")

# Newton's method
x = list(population.keys())
y = list(population.values())
ln = len(y)

coeffs = [[0] * ln for _ in range(ln)]
for i in range(1, ln):
    for j in range(ln - i):
        if (i == 1):
            coeffs[i][j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
        else:
            coeffs[i][j] = (coeffs[i - 1][j + 1] - coeffs[i - 1][j]) / (x[j + i] - x[j])

x_plot = []
for i in range(1910, 2020, 10):
    x_plot.append(i)

app_pop = [0] * len(x_plot)
for k in range(0, len(x_plot)):
    res = y[0]

    for i in range(1, ln):
        tmp_res = coeffs[i][0] 
        for j in range(1, i + 1):
            tmp_res *= (x_plot[k] - x[j - 1])
        res += tmp_res

    print(res)
    print(x_plot[k])
    app_pop[k] = res

plt.plot(x_plot, app_pop)
plt.savefig("img/newton.png")
plt.show()
