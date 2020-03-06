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
        # I replaced par_a for par_b, it worked only for square qrids
        for i in np.arange(par_b): 
            for j in np.arange(par_a):
                grid[i * par_a + j] = np.array([i, j])

        # compute distances based on coordinates
        dist = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(i+1, n):
                dist[i, j] = np.linalg.norm(grid[i, :]-grid[j, :], norm)
                dist[j, i] = dist[i, j] # distance is symetrical

        # for us sufficient sollution is 10 percent worse than f*
        self.fstar = n+np.mod(n, 2)*(2 ** (1/norm)-1)
        self.n = n # number of cities
        self.dist = dist
        # n-1 because the first city is pre-determined
        self.a = np.zeros(n-1, dtype = np.int) # only zeros
        self.b = np.arange(n-2, -1, -1) # from top to down
        self.par_a = par_a
        self.par_b = par_b

    def generate_point(self, method="method2"):
        """
        Random point generator
        :return: random point from the domain
        """
        # we are making encoded wector
        def method0(): # ----------------------------------------------------
            route = np.zeros(self.n-1)
            for i in range(0, self.par_b):
                if np.mod(i,2) == 0:
                        for j in range(0, self.par_a):
                            if i == 0 and j == 0:
                                route[0:self.par_a - 2] = 0
                            else:
                                route[i*(self.par_a) - 1 + j] = 0
                else:
                    for j in range(0, self.par_a):
                        route[i*(self.par_a) -1 + j] = self.par_a - 1 - j
            return route
            
        # this is good, but often it will generate f* sollutions, which we
        # do not want at the begining
        def method1(): # full right, one top, full left - 1, top one etc. ---
            route = np.zeros(self.n-1)
            for i in range(0, self.par_b):
                if np.mod(i,2) == 0:
                    for j in range(0, self.par_a - 1):
                        route[i*(self.par_a - 1) + j] = i
                else:
                    for j in range(0, self.par_a - 1):
                        route[i*(self.par_a - 1) + j] = i + self.par_a - 2 - j
            route[-(self.par_b - 1):] = np.arange(self.par_b - 2, -1, -1)
            return route
         
        def method2(): # random generation ----------------------------------
            route = np.array([np.random.randint(self.a[i], self.b[i] + 1) \
                             for i in np.arange(self.n-1)], dtype=int)
            return route
        
        # random choose between method0 and method2
        def method3():
            which_method = np.random.randint(0,2)
            if which_method == 0:
                route = method0()
            else:
                route = method2()
            return route
     
        return locals()[method]() # we return encoded route between cities
    
    def encode(self,x):
        """
        Encodes solution vector into ordered list of visited cities
        :param x: encoded vector, e.g.: 1 2 2 1 0
        :return:  decoded vector, e.g.: 2 4 5 3 1
        """ 
        x = x[1:] # encoded vector has one element less   
        cx = np.zeros(self.n - 1, dtype=np.int)  # the final index tour
        ux = np.ones(self.n - 1, dtype=np.int)  # used cities indices
        cum = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = np.where(cum==x[k-1])[0][0]
            cx[k-1] = ix  # append index of visited city into final index tour
            # visited city can not be included in the tour any more
            cum = np.delete(cum, ix)
        return cx

    def decode(self, x):
        """
        Decodes solution vector into ordered list of visited cities
        :param x: encoded vector, e.g.: 1 2 2 1 0
        :return:  decoded vector, e.g.: 2 4 5 3 1
        """
        cx = np.zeros(self.n, dtype=np.int)  # the final tour
        ux = np.ones(self.n, dtype=np.int)  # used cities indices
        ux[0] = 0  # first city is used automatically
        cum = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = int(x[k-1]+1)  # order index of currently visited city
            cc = cum[ix]  # currently visited city
            cx[k] = cc  # append visited city into final tour
            # visited city can not be included in the tour any more
            cum = np.delete(cum, ix)
        return cx

    def tour_dist(self, cx):
        """
        Computes a tour length
        :param cx: decoded vector
        :return: tour length
        """
        # maybe here was a bug, I changed that
        d = 0
        for i in np.arange(self.n):
            # searching in matrix
            dx = self.dist[cx[i-1], cx[i]] if i>0 else \
            self.dist[cx[self.n-1], cx[i]] # route must be connected
            d += dx # we are computing all the root, adding distances to final
        return d

    def evaluate(self, x):
        """
        Objective function evaluating function
        :param x: point
        :return: objective function value
        """
        cx = self.decode(x) # we decode vector and return distance of the tour
        return self.tour_dist(cx)

    def get_neighborhood(self, x, d=1):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        ### MAKE THIS BETTER FOR ASIGNMENT
        
        # I cannot see how I can do SWITCH, REVERSE and MOVE operations on
        # encoded vectors
        x = self.decode(x) # for better comprehention, we decode vector x
        nd = [] # our list of neighbourhoods
        # we choose one element of our vector and we will generate all
        # neighbourhoods with every other element in field from some of method
        idx = np.random.choice(np.arange(len(x)), 1, replace = False)[0]
        
        # we are going to throw a dice and decide, which method we will choose
        # for generating neighbourhood
        method = np.random.randint(1,4)
        
        # SWAP - switch two elements in vector
        if method == 1:
            for i, xi in enumerate(x):
                if i != idx:
                    xn = x.copy()
                    xn[i] = x[idx]
                    xn[idx] = x[i]
                    nd.append(xn)
                
        # REVERSE - reverse certain section of vector
        if method == 2:
            for i, xi in enumerate(x):
                xn = x.copy() # neighbourhood of x
                if i < idx:
                    reverse = xn[i:idx + 1]
                    reverse = reverse[::-1]
                    xn[i:idx + 1] = reverse
                    nd.append(xn)
                elif i > idx:
                    reverse = xn[idx:i + 1]
                    reverse = reverse[::-1]
                    xn[idx:i + 1] = reverse
                    nd.append(xn)
                
        # MOVE - move certain element of array to certain position
        if method == 3:
            for i, xi in enumerate(x):
                xn = x.copy() # neighbourhood of x
                # x = np.append(x[0], x, x[-1])
                if i < idx:
                    xn[i] = x[idx]
                    xn[i+1:idx+1] = x[i:idx]
                    nd.append(xn)
                elif i > idx:
                    xn[i-1] = x[idx]
                    xn[:i-1] = np.delete(x[:i],idx)
                    nd.append(xn)
        
        # THIS IS ORIGINAL BUT WITH ENCODED VECTOR
#        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
#        for i, xi in enumerate(x):
#            # x-lower
#            # (!) mutation correction .. will be discussed later
#            if x[i] > self.a[i]:
#                xl = x.copy()
#                xl[i] = x[i]-1
#                nd.append(xl)
#
#            # x-upper
#            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
#                xu = x.copy()
#                xu[i] = x[i]+1
#                nd.append(xu)
        
        # because of implementation of encode(), make vectors start from 0
        nd = [np.roll(n, -int(np.where(n == 0)[0][0])) for n in nd]
        # ENCODING
        nd = [self.encode(n) for n in nd]

        return nd
