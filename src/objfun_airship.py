from objfun import ObjFun
import numpy as np


class AirShip(ObjFun):

    """
    1-dimensional demo task from the first exercise
    """

    def __init__(self):
        """
        Hard-coded initialization
        """
        fstar = -100
        a = 0
        b = 799
        super().__init__(fstar, a, b)

    def generate_point(self):
        """
        Random point generator
        :return: random point from the domain
        """
        return np.random.randint(0, 800)

    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        left = [x for x in np.arange(x-1, x - d - 1, -1, dtype=int) if x >= 0]
        right = [x for x in np.arange(x+1, x + d + 1, dtype=int) if x < 800]
        if np.size(left) == 0:
            return right
        elif np.size(right) == 0:
            return left
        else:
            return np.concatenate((left, right))

    def evaluate(self, x):
        """
        Objective function evaluating function
        :param x: point
        :return: objective function value
        """
        px = np.array([0,  50, 100, 300, 400, 700, 799], dtype=int)
        py = np.array([0, 100,   0,   0,  25,   0,  50], dtype=int)
        xx = np.arange(0, 800)
        yy = np.interp(xx, px, py)
        return -yy[x]  # negative altitude, because we are minimizing (to be consistent with other obj. functions)
