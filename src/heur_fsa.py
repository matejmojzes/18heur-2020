from heur import Heuristic, StopCriterion
import numpy as np


class FastSimulatedAnnealing(Heuristic):

    """
    Implementation of Fast Simulated Annealing heuristic
    """

    def __init__(self, of, maxeval, T0, n0, alpha, mutation):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param T0: initial temperature
        :param n0: cooling strategy parameter - number of steps
        :param alpha: cooling strategy parameter - exponent
        :param mutation: mutation to be used for the specific objective function (see heur_aux.py)
        """
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.mutation = mutation

        self.name = 'FSA' + self.mutation.get_name()

    def get_specs(self):
        return self.get_name() + '_T0={}_n0={}_alpha={}'.format(self.T0, self.n0, self.alpha)

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        try:
            x = self.of.generate_point()
            f_x = self.evaluate(x)
            while True:
                k = self.neval - 1  # because of the first obj. fun. evaluation
                t0 = self.T0
                n0 = self.n0
                alpha = self.alpha
                t = t0 / (1 + (k / n0) ** alpha) if alpha > 0 else t0 * np.exp(-(k / n0) ** -alpha)

                y = self.mutation.mutate(x)
                f_y = self.evaluate(y)
                s = (f_x - f_y)/t
                swap = np.random.uniform() < 1/2 + np.arctan(s)/np.pi
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, 'T': t, 'swap': swap})
                if swap:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise
