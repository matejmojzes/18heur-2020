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
        self.name = 'cor'

    def get_name(self):
        return self.name

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)


class MirrorCorrection(Correction):
    """
    Mutation correction via mirroring
    """

    def __init__(self, of):
        Correction.__init__(self, of)
        self.name = 'mir'

    def correct(self, x):
        n = np.size(x)
        d = self.of.b - self.of.a
        for k in range(n):
            if d[k] == 0:
                x[k] = self.of.a[k]
            else:
                de = np.mod(x[k] - self.of.a[k], 2*d[k])
                de = np.amin([de, 2*d[k] - de])
                x[k] = self.of.a[k] + de
        return x


class ExtensionCorrection(Correction):
    """
    Mutation correction via periodic domain extension
    """

    def __init__(self, of):
        Correction.__init__(self, of)
        self.name = 'ext'

    def correct(self, x):
        d = self.of.b - self.of.a
        x = self.of.a + np.mod(x - self.of.a, d + (1 if is_integer(x) else 0))
        return x


class Mutation:

    """
    Generic mutation super-class
    """

    def __init__(self, correction):
        self.correction = correction
        self.name = 'abstract'

    def get_name(self):
        return self.name

    def mutate(self, x):
        raise NotImplementedError("Mutation must implement its own mutate function")


class CauchyMutation(Mutation):

    """
    Cauchy mutation
    """

    def __init__(self, r, correction):
        Mutation.__init__(self, correction)
        self.r = r
        self.name = 'cau_{}_r={}'.format(correction.get_name(), self.r)

    def mutate(self, x):
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r * np.tan(np.pi * (u - 1 / 2))
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)  # optional rounding
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected
