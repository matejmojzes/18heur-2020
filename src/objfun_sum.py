from objfun import ObjFun
import numpy as np


class Sum(ObjFun):

    """
    sum(x): trivial objective function for demo purposes
    """

    def __init__(self, a, b):
        """
        Initialization (domain specification)
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        self.n = np.size(a)  # dimension of the task
        name = 'sum'
        super().__init__(fstar=0, a=a, b=b, name=name)

    def generate_point(self):
        """
        Random point generator
        :return: random point from the domain
        """
        return [np.random.randint(self.a[i], self.b[i]+1) for i in np.arange(self.n)]

    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        assert d == 1, "Sum(x) supports neighbourhood with diameter = 1 only"
        nd = []
        for i in np.arange(self.n):
            if x[i] > self.a[i]:
                xx = x.copy()
                xx[i] -= 1
                nd.append(xx)
            if x[i] < self.b[i]:
                xx = x.copy()
                xx[i] += 1
                nd.append(xx)
        return nd

    def evaluate(self, x):
        """
        Objective function evaluating function
        :param x: point
        :return: objective function value
        """
        return np.sum(x)
