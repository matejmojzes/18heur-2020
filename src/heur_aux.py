import numpy as np


def is_integer(a):
    """
    Tests if `a` is integer
    """
    dt = a.dtype
    return dt == np.int16 or dt == np.int32 or dt == np.int64


class Correction:

    """
    Baseline mutation correction strategy - "sticks" the solution to domain boundaries
    """

    def __init__(self, of):
        self.of = of

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)


class Mutation:

    """
    Generic mutation super-class
    """

    def __init__(self, correction):
        self.correction = correction


class CauchyMutation(Mutation):

    """
    Cauchy mutation
    """

    def __init__(self, r, correction):
        Mutation.__init__(self, correction)
        self.r = r

    def mutate(self, x):
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r * np.tan(np.pi * (u - 1 / 2))
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)  # optional rounding
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected
