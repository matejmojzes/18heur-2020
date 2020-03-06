from heur import Heuristic, StopCriterion
from heur_aux import CauchyMutation, MirrorCorrection
import numpy as np


class FastSimulatedAnnealing(Heuristic):

    """
    Implementation of Fast Simulated Annealing heuristic
    """

    def __init__(self, of, T0, maxeval, n0, alpha):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param T0: initial temperature
        :param n0: cooling strategy parameter - number of steps
        :param alpha: cooling strategy parameter - exponent
        :param mutation: mutation to be used for the specific objective
        function (see heur_aux.py)
        """
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.mutation = CauchyMutation(r = 0.5, correction = 
                                       MirrorCorrection(self.of))

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        try: # we will ececute this until we get exeption in Heuristic class
            x = self.of.generate_point() # only once we generate poin and eval
            f_x = self.evaluate(x)
            while True:
                k = self.neval - 1  # because of the first obj. fun. evaluation
                T0 = self.T0
                n0 = self.n0
                alpha = self.alpha
                T = T0 / (1 + (k / n0) ** alpha) if alpha > 0 else T0 * \
                np.exp(-(k / n0) ** -alpha) # update temperature - k is changed
                
                y = self.mutation.mutate(x) # new vector
                f_y = self.evaluate(y) # what is the cost at new vector?
                s = (f_x - f_y)/T # decision criteria for swaping
                swap = np.random.uniform() < 1/2 + np.arctan(s)/np.pi # T/False
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, \
                          'T': T, 'swap': swap})
                if swap: # we change our point
                    x = y
                    f_x = f_y

        except StopCriterion:
            
            # CODED FOR ASSIGNMENT
            if self.neval < np.inf: # if we reached best, we need to log solut.
                # we are out of evals, go go directly to objfun and eval method
                try:
                    f_y = self.of.evaluate(y)
                except: # case that we are lucky and we generate f* sol. first
                    f_y = self.of.evaluate(x)
                    y = x
                    k = 1
                    T = self.T0
                # Because it finished before we reachend f* solution and we
                # know that we got better.
                swap = True
                if swap: # we change last line
                    x = y
                    f_x = f_y
                # x and y will be the same in our last row and also f_x and f_y
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, \
                          'T': T, 'swap': swap})
            return self.report_end()
        except:
            raise
