from forbiddenfruit import curse
import numpy as np

def cursed(pyclass):
    def cursed_decorator(func):
        curse(pyclass, func.__name__, func)
        return func
    return cursed_decorator

@cursed(range)
def pipeline(self):
    return self

@cursed(range)
def unroll(self, factor=None):
    return self

@cursed(range)
def partition(self, factor=None):
    return self

@cursed(np.ndarray)
def array_partition(self, factor=None, dim=None):
    return self

def pragma(str):
    pass
