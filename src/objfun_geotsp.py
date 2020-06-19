"""Class to define Travelling Salesman Problem with places holding real geospatial coordinates."""
from objfun import ObjFun
import numpy as np
from datetime import datetime
from os.path import join
import googlemaps


class GeoTSP(ObjFun):

    def __init__(self, spots, key=None, dm_path=None, max_rows=None):
        """
        Initialization
        :param spots: list of addresses
        :param key: API key for the google APIs (Roads, Distance Matrix, Maps Javascript, Directions)
        :param dm_path: Path to the csv file with the saved duration and distance matrix
        :param max_rows: Maximum number of rows from 'spots' to be read
        """

        self.fstar = 0  # Theoretical minimum
        self.spots = spots
        self.n = len(spots)  # Number of rows in the array
        if max_rows and max_rows < self.n:
            # If the max_rows parameter is specified and is less than the number of
            # the possible locations in the spot file, then assign it to the self.n
            self.n = max_rows
        self.a = np.zeros(self.n-1, dtype=np.int)  # n-1 because the first city is pre-determined
        self.b = np.arange(self.n-2, -1, -1)
        if dm_path is not None:
            dim_path = join(dm_path, "distance_matrix.csv")
            dum_path = join(dm_path, "duration_matrix.csv")
            self.dist_matrix = self.get_matrix(dm_path=dim_path)
            self.dura_matrix = self.get_matrix(dm_path=dum_path, mtype="duration")
        else:
            self.dist_matrix = self.get_matrix(key=key)
            self.dura_matrix = self.get_matrix(key=key, mtype="duration")


        #if a_matrix is None:
        #    a_matrix = np.ones([self.n, self.n], int)
        #    self.a_matrix = np.fill_diagonal(a_matrix, 0)  # zeros on the diagonal
        #else:
        #    # TODO: Structure checks
        #    self.a_matrix = a_matrix


    def get_matrix(self, key=None, dm_path=None, mtype="distance"):
        """
        Computes distance matrix using googlemaps module.
        :param key: Google API key
        :param dm_path: Path to the csv file with distance matrix
        :param mtype: Either "distance" or "duration"
        :return: Distance or Duration matrix (Distance by default)
        """

        if dm_path is not None:
            cols = [i for i in range(0, self.n)]
            return np.genfromtxt(dm_path, delimiter=',', max_rows=self.n, usecols=cols)
        elif not key:
            print("No key or dm_path specified!")
            raise Exception

        # Must set a departure time when using mode="driving"
        now = datetime.now()
        # Google Maps Client needs to be set with the API key
        gmaps = googlemaps.Client(key)

        # API will only accept a combined total of 25 Origins and Destinations
        # We will iterate every time with 1 destination and 'n' origins split to the chunks with length of 24 at max
        spots_chunks = list(self.chunks(24))
        matrix = np.zeros([self.n, self.n])

        for n in range(self.n):  # For each spot
            abs_spot_number = 0  # Absolute position number of the spot in the row
            for m in range(len(spots_chunks)):  # For each chunk in the spots list
                dm = gmaps.distance_matrix(self.spots[n], spots_chunks[m], mode="driving", units="metric",
                                           departure_time=now)
                for k in range(len(spots_chunks[m])):  # Compute distance for each element in the m-th chunk
                    #  needs to move with chunks
                    matrix[n][abs_spot_number] = dm["rows"][0]["elements"][k][mtype]['value']
                    abs_spot_number += 1

        return matrix


    def chunks(self, n):
        """Yield successive n-sized chunks from list."""
        for i in range(0, len(self.spots), n):
            yield self.spots[i:i + n]


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


    def tour_dist(self, cx, mtype="distance"):
        """
        Computes a tour length
        :param cx: decoded vector
        :param mtype: Duration or Distance matrix
        :return: tour length
        """
        d = 0
        matrix = self.dura_matrix if mtype == "duration" else self.dist_matrix
        for i in np.arange(self.n - 1):  # Doing cumsum => one step less than the number of all of the elements
            d += matrix[cx[i], cx[i+1]]
        return d


    def evaluate(self, x, mtype="distance"):
        """
        Objective function evaluating function
        :param x: point
        :param mtype: Duration or Distance
        :return: objective function value
        """
        cx = self.decode(x)
        return self.tour_dist(cx, mtype=mtype)


    def get_neighborhood(self, x, d=1):
        """
        Solution neighborhood generating function
        :param x: spot number
        :param d: diameter of the neighbourhood
        """

        assert d == 1, "GeoTSP supports neighborhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a[i]:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i] - 1
                nd.append(xl)

            # x-upper
            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i] + 1
                nd.append(xu)

        return nd


    def find_k_nn(self, spot_number, k, matrix="distance"):
        """
        Finds 'k' nearest neighbours of the location identified by 'spot_number'
        :param spot_number: 0 - (n-1), position of the spot
        :param matrix: consider 'distance' or 'duration' as a measure
        :param k: number of nearest neighbours
        :return: list of 'k' nearest neighbours of the spot with the 'spot_number'

        Note:
            TODO: Take the advantage of the k-nn p.e. in the get_neighborhood() function.
        """
        assert k < self.n, "The number of neighbours has to be smaller than 'n' !"
        if matrix == "distance":
            row = self.dist_matrix[spot_number, :]
        elif matrix == "duration":
            row = self.dura_matrix[spot_number, :]
        else:
            print("Wrong option for measure matrix! Only 'distance' or 'duration' are allowed.")
            return

        # Indexes of 'k' nearest neighbours.
        # not taking the first one, since it'll always be the same place as the
        # place from where we are looking for neighbors
        idx = np.argpartition(row, list(range(0, k+1)))[1:k+1]
        return idx


    def get_route(self, x):
        """
        Prints string with the sequence of the names in the correct order to be reached in.
        :param x: encoded point
        :return: string with the route
        """
        cx = self.decode(x)
        route = ""
        for i in cx[:-1]:
            route += f"{self.spots[i]} -> \n"
        return route + str(self.spots[cx[-1]])
