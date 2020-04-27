from objfun import ObjFun
import numpy as np


class DeJong1(ObjFun):
    """
    De Jong function 1 (sphere)
    According to http://www.geatbx.com/docu/fcnindex-01.html#P89_3085
    """

    def __init__(self, n, eps=0.01):
        self.fstar = 0 + eps
        self.n = n
        self.a = -5.12*np.ones(self.n, dtype=np.float64)
        self.b = 5.12*np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return np.sum([x[i] ** 2 for i in range(self.n)])
