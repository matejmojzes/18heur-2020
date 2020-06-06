from objfun import ObjFun
import numpy as np


class Plato(ObjFun):
    """
    Rastrigin's function 6
    According to http://www.geatbx.com/docu/fcnindex-01.html#P140_6155
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
        self.a = -5.12*np.ones(self.n, dtype=np.float64)
        self.b = 5.12*np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return 10*self.n + np.sum([x[i] ** 2 - 10*np.cos(2*np.pi*x[i]) for i in range(self.n)])

    def get_n(self):
        return self.n


