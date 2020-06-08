from heur import Heuristic, StopCriterion
import numpy as np
from random import shuffle

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
        :param random_descent: turns on random descent, instead of the
        steepest one (default)
        """
        Heuristic.__init__(self, of, maxeval)
        self.hmax = hmax
        self.random_descent = random_descent

    def steepest_descent(self, x):
        """
        Steepest/Random Hill Descent
        :param x: beginning point
        """
        desc_best_y = np.inf
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax: # go=TRUE and number of evaluations < hmax
            go = False # we assume, that we will not improve by grad descent
            h += 1 # we increase our evaluation count by 1

            "we are looking for neighborhood around our current point"
            neighborhood = self.of.get_neighborhood(desc_best_x)
            if self.random_descent: # if we use random descent for optimization
                try:
                    np.random.shuffle(neighborhood) # we randomly choose new point
                except:
                    shuffle(neighborhood)

            "we are using steepest descent for optimization"
            for xn in neighborhood: # we go through all neighborhoods
                # call evaluate method of Heuristic class, which call evaluate
                # of ObjFun, argument is one point of neighbour
                yn = self.evaluate(xn)
                if yn < desc_best_y: # if we are better
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                    """"
                    if random_descent, we firstly shuffle neighbourhood and
                    then we take firs.....before we execute one evaluation of
                    random_descent and then we have to end the for loop
                    """
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
