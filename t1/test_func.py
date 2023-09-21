import math
from decimal import Decimal

def func1(val):
    return math.sin(val ** 2)

def func1_der(val):
    return 2 * val * math.cos(val ** 2)

def func2(val):
    return math.cos(math.sin(val))

def func2_der(val):
    return -math.cos(val) * math.sin(math.sin(val))

def func3(val):
    return math.exp(math.sin(math.cos(val)))

def func3_der(val):
    return func3(val) * (- math.sin(val) * math.cos(math.cos(val)))

def func4(val):
    return math.log(val + 3)

def func4_der(val):
    return 1 / (val + 3)

def func5(val):
    return (val + 3) ** 0.5

def func5_der(val):
    return 1 / (2 * func5(val))

class TestFunc:
    def __init__(self, func, func_deriv):
        self.func = func
        self.deriv = func_deriv
        
class TestFuncData:
    def __init__(self):
        self.data = [TestFunc(func1, func1_der), TestFunc(func2, func2_der), TestFunc(func3, func3_der),
                TestFunc(func4, func4_der), TestFunc(func5, func5_der)]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        self.index = self.index + 1
        return self.data[self.index - 1]
