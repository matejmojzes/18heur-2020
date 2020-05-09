from heur import Heuristic, StopCriterion
import numpy as np


class ShootAndGo(Heuristic):

    """
    Implementation of generalized Shoot & Go heuristic
    """

    def __init__(self, of, maxeval, hmax=np.inf, random_descent=False):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param hmax: maximum number of local improvements (0 = Random Shooting)
        :param random_descent: turns on random descent, instead of the steepest one (default)
        """
        Heuristic.__init__(self, of, maxeval)
        self.hmax = hmax
        self.random_descent = random_descent

        if self.random_descent:
            self.name = 'SG_RD'
        else:
            self.name = 'SG_SD'
        if self.hmax is 0:
            self.name = 'SG_RS'

    def get_specs(self):
        return self.get_name() + '_hmax={}'.format(self.hmax)

    def steepest_descent(self, x):
        """
        Steepest/Random Hill Descent
        :param x: beginning point
        """
        desc_best_y = np.inf
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax:
            go = False
            h += 1

            neighborhood = self.of.get_neighborhood(desc_best_x, 1)
            if self.random_descent:
                np.random.shuffle(neighborhood)

            for xn in neighborhood:
                yn = self.evaluate(xn)
                if yn < desc_best_y:
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                    if self.random_descent:
                        break

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        try:
            while True:
                # Shoot...
                x = self.of.generate_point()  # global search
                self.evaluate(x)
                # ...and Go
                if self.hmax > 0:
                    self.steepest_descent(x)  # local search (optional)

        except StopCriterion:
            return self.report_end()
        except:
            raise
