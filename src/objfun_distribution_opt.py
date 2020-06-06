from objfun import ObjFun
import numpy as np


class VolumeDistribution(ObjFun):

    """
    fitness of distributions, i.e. x\in R^{n} and sum(x) = 1
    """

    def __init__(self, d, delta, alphas, fstar=-np.inf):
        # Prepare the alphas to matrix shape
        self.alphas = [[i * delta + a_i["r"] - a_i["m"] if i * delta <= a_i["m"] else 0 for i in range(0, d)] for a_i in alphas]  # The setup parameters
        self.alphas = np.reshape(self.alphas, (len(alphas), d))
        self.fstar = fstar  # depends on the setup of alphas
        self.delta = delta  # the value step in the discrete distribution
        self.n = d  # the number of steps in distribution, i.e. support = d * delta
        self.a = np.zeros(self.n, dtype=float)  # this is very problematic task
        self.b = np.ones(self.n, dtype=float)

    def generate_point(self):
        res = np.random.rand(self.n)
        res /= sum(res)
        return res

    def get_neighborhood(self, x, d):
        """
        Also very problematic
        :param: x
        """
        """
        assert d == 1, "Zebra3 supports neighbourhood with (Hamming) distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xx = x.copy()
            xx[i] = 0 if xi == 1 else 1
            nd.append(xx)
        return nd
        """
        return x


    def evaluate(self, x):
        """
        Evaluation of fitness of a single distribution
        """
        return -np.sum(np.multiply(self.alphas, x))



class VolumeDistributionPenalization(ObjFun):

    """
    Distribution with penalization for the sum not equal to 1
    """

    def __init__(self, d, delta, alphas, fstar=-np.inf, penalization = 1):
        # Prepare the alphas to matrix shape
        self.alphas = [[i * delta + a_i["r"] - a_i["m"] if i * delta <= a_i["m"] else 0 for i in range(0, d)] for a_i in alphas]  # The setup parameters
        self.alphas = np.reshape(self.alphas, (len(alphas), d))
        self.fstar = fstar  # depends on the setup of alphas
        self.delta = delta  # the value step in the discrete distribution
        self.penalization = penalization  # The penalization parameter
        self.n = d  # the number of steps in distribution, i.e. support = d * delta
        self.a = np.zeros(self.n, dtype=float)  # this is very problematic task
        self.b = np.ones(self.n, dtype=float)

    def generate_point(self):
        res = np.random.rand(self.n)
        res /= sum(res)
        return res

    def get_neighborhood(self, x, d):
        """
        Also very problematic
        :param: x
        """
        """
        assert d == 1, "Zebra3 supports neighbourhood with (Hamming) distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xx = x.copy()
            xx[i] = 0 if xi == 1 else 1
            nd.append(xx)
        return nd
        """
        return x


    def evaluate(self, x):
        """
        Evaluation of fitness of a single distribution
        """
        p = abs(sum(x) - 1) * self.alphas.shape[0] * self.alphas.shape[1] * self.penalization
        return -np.sum(np.multiply(self.alphas, x)) + p

def main_test():

    alphas = [{"r": 15, "m": 10},{"r": 12, "m": 9.5}]
    delta = 1
    d = 15
    problem = VolumeDistributionPenalization(d, delta, alphas)
    print(problem.alphas)

    x = problem.generate_point()
    x = np.reshape([0.1 if i != 10 else 1 for i in range(0,d)], d)

    print(x)

    x[8] = 0.5
    print(x)
    print(problem.evaluate(x))

if __name__ == "__main__":
	main_test()