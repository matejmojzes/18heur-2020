import numpy as np


def is_integer(a):
    """
    Tests if `a` is integer.
    Takes a np.array
    """
    for i, n in enumerate(a):
        if is_number(n):
            if round(n) != n:
                return False
    return True

def is_number(a):
    """
    Tests if `a` is number.
    Takes one element.
    """
    dt = type(a)
    return (dt == int or dt == np.int16 or dt == np.int32 or dt == np.int64
            or dt == float or dt == np.float16 or dt == np.float32 or
            dt == np.float64)

class Correction:

    """
    Baseline mutation correction strategy - "sticks" the solution to domain
    boundaries
    """

    def __init__(self, of):
        self.of = of

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)
        "when we are out of boundaries"


class MirrorCorrection(Correction):
    """
    Mutation correction via mirroring
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        n = np.size(x)
        d = self.of.b - self.of.a
        for k in range(n):
            if d[k] == 0: # if a[k] is the same as b[k], there is no space
                x[k] = self.of.a[k]
            else:
                de = np.mod(x[k] - self.of.a[k], 2*d[k]) #de is less than 2d[k]
                de = np.amin([de, 2*d[k] - de]) # now de is less than d[k]
                x[k] = self.of.a[k] + de # we are sure that we will be in <a,b>
        return x


class ExtensionCorrection(Correction):
    """
    Mutation correction via periodic domain extension
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        d = self.of.b - self.of.a
        " +1 causes that we can get new x in b "
        x = self.of.a + np.mod(x - self.of.a, d + (1 if is_integer(x) else 0))
        return x
    
class HyperparametersCorrection(Correction):
    """
    Special class for TuneHyperparameters objective function
    """
    
    def __init__(self, of, discrete):
        Correction.__init__(self, of)
        self.discrete = discrete
    
    # maybe we will have to change this, because we are looking
    def correct(self, x, a, b):
        j = 0 # index for dicrete list
        for i, n in enumerate(x):
            if is_number(n):
               difference = b[i] - a[i]
               x[i] = a[i] + np.mod(x[i] - a[i], difference
                + (1 if self.discrete[j] else 0))
            if self.discrete[j]:
                x[i] = round(x[i])
                j = j + 1
        return x


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
        x_new = x + r * np.tan(np.pi * (u - 1 / 2)) # move to new point
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)  # optional rounding
        " we will get a lot of boudary values - in assignment we can improve"
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected
