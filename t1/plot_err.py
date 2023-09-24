import math
import matplotlib.pyplot as plt

import test_func as tests

def deriv1(x, h, func):
    return (func(x + h) - func(x)) / h

def deriv2(x, h, func):
    return (func(x) - func(x - h)) / h

def deriv3(x, h, func):
    return (func(x + h) - func(x - h)) / (2 * h)

def deriv4(x, h, func):
    return (4 * (func(x + h) - func(x - h)) / ( 3 * 2 * h) - 
            (func(x + 2 * h) - func(x - 2 * h)) / ( 3 * 4 * h))

def deriv5(x, h, func):
    return (3 * (func(x + h) - func(x - h)) / ( 2 * 2 * h) -  3 * 
            (func(x + 2 * h) - func(x - 2 * h)) / ( 5 * 4 * h) + 
            (func(x + 3 * h) - func(x - 3 * h)) / ( 10 * 6 * h))

ApproxDerivs = [deriv1, deriv2, deriv3, deriv4, deriv5]

def CalcDelta(pwr):
    return 2 ** (1 - pwr)

def SinglePlotData(testf, derf, point, maxpwr):
    xdata = []
    ydata = []
    for i in range(maxpwr):
        xdata.append(math.log(CalcDelta(i + 1)))
        ydata.append(math.log(abs(derf(point, CalcDelta(i + 1), testf.func) - testf.deriv(point))))
    return xdata, ydata

test_samples = tests.TestFuncData()
iter(test_samples)

for j, tstf in enumerate(test_samples):
    plt.figure(figsize=(9, 5))

    for i in range(len(ApproxDerivs)):
        x, y = SinglePlotData(tstf, ApproxDerivs[i], 8, 21)
        plt.plot(x, y, label = "Appr. function " + str(i + 1))

    plt.legend()
    plt.title("Test sample № " + str(j + 1))
    plt.show()
