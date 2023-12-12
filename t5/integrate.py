import matplotlib
import math
import numpy as np

# VII.9.5(a)
step = 0.25
fx = np.array([1.000000, 0.989616, 0.958851, 0.908852, 0.841471, 0.759188, 0.664997, 0.562278, 0.454649])

def trapeze_int(step, fx):
    fx_sum = sum(fx)
    first_and_last_avg = (fx[0] + fx[-1]) / 2
    result = (fx_sum - first_and_last_avg) * step
    return result

def simpson_int(step, fx):
    odd  = 0
    even = 0

    for i in range(1, len(fx) - 1):
        if i % 2 == 0:
            even += fx[i]
        else:
            odd += fx[i]

    first_term = fx[0]
    last_term = fx[-1]
    odd_sum = 4 * odd
    even_sum = 2 * even
    result = (first_term + odd_sum + even_sum + last_term) * step / 3
    return result

def richardson(raw_int, st):
    return raw_int(step, fx) + (raw_int(step, fx) - raw_int(step * 2, fx[::2])) / (2 ** st - 1)

print('===    integration results    ====')
print('Simpson: ', simpson_int(step, fx))
print('Trapeze: ', trapeze_int(step, fx))
print('Ruchardson(trapeze): ', richardson(trapeze_int, 2))
print('Ruchardson(simpson): ', richardson(simpson_int, 4))
print('Real(wolfram):        1.6054129768026948')
