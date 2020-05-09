from objfun import ObjFun
import numpy as np


class HMeans(ObjFun):
    """
    Heuristic clustering inspired by k-means
    """

    def __init__(self, dim=2, n_clu=3, n_pts=15, eps=0.1):
        """
        Pseudo-randomly generates cluster centroids and data point matrix
        :param dim: dimension
        :param n_clu: number of clusters
        :param n_pts: number of data points per cluster
        :param eps: tolerance from f*
        """
        np.random.seed(10)
        C = np.random.uniform(size=(n_clu, dim))  # cluster centers

        # generate points for each cluster
        X = np.zeros((0, dim))  # data point matrix
        sigma = 0.05
        for c in C:
            P = np.zeros((n_pts, dim))
            for i in range(n_pts):
                P[i] = [np.random.normal(cc, sigma) for cc in c]
            X = np.concatenate((X, P), axis=0)

        self.n_clu = n_clu
        self.dim = dim
        a = X.min(axis=0)
        self.a = np.concatenate([a for i in range(n_clu)])  # repeat for each solution centroid
        b = X.max(axis=0)
        self.b = np.concatenate([b for i in range(n_clu)])  # -- // --
        self.X = X
        self.C = C
        x_centroids = self.encode_solution(C)
        self.fstar = self.evaluate(x_centroids) + eps
        self.name = 'hmeans'

    def encode_solution(self, C):
        """
        Encodes centroids into solution usable by heuristics
        :param C: matrix with centroids
        :return: array
        """
        return C.flatten()

    def decode_solution(self, x):
        """
        Decodes heuristic solution into centroids
        :param x: array
        :return: matrix with centroids
        """
        return np.reshape(x, (self.n_clu, self.dim))

    def generate_point(self):
        C = np.random.uniform(size=(self.n_clu, self.dim))  # randomly generates centroids
        return self.encode_solution(C)

    def evaluate(self, x):
        """
        Computes sum of squares of distances from data points to their nearest centroids
        """
        ssq = 0
        C = self.decode_solution(x)
        for x in self.X:  # for each data point:
            d = np.array([np.linalg.norm(c - x) for c in C])  # compute distances to centroids
            ix = d.argmin()  # index of the nearest centroid
            ssq += d[ix] ** 2  # add square of distance to the nearest centroid
        return ssq

    def get_cluster_labels(self, x):
        """
        Returns array with cluster labels [0; n_clu] for a given solution vector
        """
        C = self.decode_solution(x)
        labels = np.zeros(self.X.shape[0], dtype=int)
        for i, x in enumerate(self.X):  # for each data point:
            d = np.array([np.linalg.norm(c - x) for c in C])  # compute distances to centroids
            ix = d.argmin()  # index of the nearest centroid
            labels[i] = ix
        return labels
