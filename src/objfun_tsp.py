from objfun import ObjFun
import numpy as np


class TSPGrid(ObjFun):

    def __init__(self, par_a, par_b, norm=2):
        """
        Initialization
        :param par_a: width of grid
        :param par_b: height of grid
        """
        n = par_a * par_b  # number of cities

        # compute city coordinates
        grid = np.zeros((n, 2), dtype=np.int)
        for i in np.arange(par_a):
            for j in np.arange(par_b):
                grid[i * par_b + j] = np.array([i, j])

        # compute distances based on coordinates
        dist = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(i+1, n):
                dist[i, j] = np.linalg.norm(grid[i, :]-grid[j, :], norm)
                dist[j, i] = dist[i, j]

        self.fstar = n+np.mod(n, 2)*(2 ** (1/norm)-1)
        self.n = n
        self.dist = dist
        self.a = np.zeros(n-1, dtype=np.int)  # n-1 because the first city is pre-determined
        self.b = np.arange(n-2, -1, -1)

    def generate_point(self):
        """
        Random point generator
        :return: random point from the domain
        """
        return np.array([np.random.randint(self.a[i], self.b[i] + 1) for i in np.arange(self.n-1)], dtype=int)

    def decode(self, x):
        """
        Decodes solution vector into ordered list of visited cities
        :param x: encoded vector, e.g.: 1 2 2 1 0
        :return:  decoded vector, e.g.: 2 4 5 3 1
        """
        cx = np.zeros(self.n, dtype=np.int)  # the final tour
        ux = np.ones(self.n, dtype=np.int)  # used cities indices
        ux[0] = 0  # first city is used automatically
        c = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = x[k-1]+1  # order index of currently visited city
            cc = c[ix]  # currently visited city
            cx[k] = cc  # append visited city into final tour
            c = np.delete(c, ix)  # visited city can not be included in the tour any more
        return cx

    def tour_dist(self, cx):
        """
        Computes a tour length
        :param cx: decoded vector
        :return: tour length
        """
        d = 0
        for i in np.arange(self.n):
            dx = self.dist[cx[i-1], cx[i]] if i>0 else self.dist[cx[self.n-1], cx[i]]
            d += dx
        return d

    def evaluate(self, x):
        """
        Objective function evaluating function
        :param x: point
        :return: objective function value
        """
        cx = self.decode(x)
        return self.tour_dist(cx)

    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a[i]:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i]-1
                nd.append(xl)

            # x-upper
            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i]+1
                nd.append(xu)

        return nd
