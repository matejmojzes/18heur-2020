from objfun import ObjFun
import numpy as np


class Zebra3(ObjFun):

    """
    Clerc's Zebra3 objective function
    See classes/20200421_Genetic_Optimization.ipynb notebook for more information
    """

    def __init__(self, d):
        self.fstar = 0
        self.d = d
        self.n = d*3
        self.a = np.zeros(self.n, dtype=int)
        self.b = np.ones(self.n, dtype=int)

    def generate_point(self):
        return np.random.randint(0, 1+1, self.n)

    def get_neighborhood(self, x, d):
        assert d == 1, "Zebra3 supports neighbourhood with (Hamming) distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xx = x.copy()
            xx[i] = 0 if xi == 1 else 1
            nd.append(xx)
        return nd

    def evaluate(self, x):
        f = 0
        for i in np.arange(1, self.d+1):
            xr = x[(i-1)*3:i*3]
            s = np.sum(xr)
            if np.mod(i,2) == 0:
                if s == 0:
                    f += 0.9
                elif s == 1:
                    f += 0.6
                elif s == 2:
                    f += 0.3
                else:
                    f += 1.0
            else:
                if s == 0:
                    f += 1.0
                elif s == 1:
                    f += 0.3
                elif s == 2:
                    f += 0.6
                else:
                    f += 0.9
        f = self.n/3-f
        return f
