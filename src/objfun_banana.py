from objfun import ObjFun
import numpy as np


class Banana(ObjFun):
    """
    Rosenbrock's valley (De Jong's function 2)
    According to http://www.geatbx.com/docu/fcnindex-01.html#P129_5426
    """

    def __init__(self, n, eps=0.01):
        """
        Default initialization function that sets:
        :param fstar: f^* value to be reached (can be -inf)
        :param n: function dimension
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        self.fstar = 0 + eps
        self.n = n
        self.a = -2.048*np.ones(self.n, dtype=np.float64)
        self.b = 2.048*np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return np.sum([100*(x[i+1] - x[i] ** 2)**2 + (1 - x[i])**2 for i in range(self.n-1)])

    def get_n(self):
        return self.n


