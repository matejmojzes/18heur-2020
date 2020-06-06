from heur import Heuristic, StopCriterion
import numpy as np
import pandas as pd


class Draset(Heuristic):

    def __init__(self, of, maxeval, E, alfa0, N_general, N_local, correction):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param E: maximum number of epochs, i.e. maximum number of the random vector generation
        :param alfa0: the first dx is generated from range [-alfa0, alfa0]
        :param N_general: alfa doesn't change for N_general steps during global search
        :param N_local: alfa doesn't change for N_local steps during local search
        :param correction: correction for x values
        """

        Heuristic.__init__(self, of, maxeval)
        self.k = 0      # alfa counter - it is in the article, but I don't use it
        self.epoch = 0  # epoch counter
        self.E = E
        self.N_general = N_general
        self.GSE = self.E - 2*self.N_general    # general search stop criterion,
                                                # number of epochs during general search
        self.N_local = N_local
        self.alfa = [alfa0]     # we will save all alpha values
        self.x_curr = None      # current value of x
        self.alfa_best = None   # alpha connected with the best solution
        self.y_gen = np.zeros(self.E+1, dtype=float)    # we save all y values - it is needed for alpha calculation
                                                        # +1 because of the initial value
        self.correction = correction

    def generate_dx(self, alfa):
        """
        Generation of a random vector from range [-alfa, alfa]
        """
        return np.random.uniform(-np.abs(alfa), np.abs(alfa), size=self.of.get_n())

    def general_search(self):
        """
        General search phase of the algorithm
        """
        while self.epoch < self.GSE:
            n = 0
            while n < self.N_general:
                dx = self.generate_dx(self.alfa[-1])
                self.epoch = self.epoch + 1
                if self.epoch > self.GSE:
                    self.epoch = self.epoch - 1
                    break
                y_new = self.evaluate(self.x_curr + dx)
                positive = False
                if y_new < self.best_y:
                    self.best_y = y_new
                    self.best_x = self.x_curr + dx
                    self.alfa_best = self.alfa[-1]
                    self.y_gen[self.epoch] = self.y_gen[self.epoch-1]
                    n = n + 1
                    positive = True
                elif y_new < self.y_gen[self.epoch-1]:
                    self.y_gen[self.epoch] = y_new
                    self.x_curr = self.x_curr + dx
                    n = n + 1
                    positive = True
                if not positive:
                    y_new = self.evaluate(self.x_curr - dx)
                    if y_new < self.best_y:
                        self.best_y = y_new
                        self.best_x = self.x_curr - dx
                        self.alfa_best = self.alfa[-1]
                        self.y_gen[self.epoch] = self.y_gen[self.epoch-1]
                        n = n + 1
                    elif y_new < self.y_gen[self.epoch-1]:
                        self.y_gen[self.epoch] = y_new
                        self.x_curr = self.x_curr - dx
                        n = n + 1
                    else:
                        self.y_gen[self.epoch] = self.y_gen[self.epoch - 1]
                self.log({'epoch': self.epoch, 'n': n, 'alfa': self.alfa[-1], 'y_new': y_new, 'y_curr': self.y_gen[self.epoch-1], 'y_best': self.best_y, 'x_best': self.best_x})
            self.k = self.k + 1
            alfa_new = self.alfa[-1]/y_new * self.y_gen[self.epoch]/self.y_gen[self.epoch-2]
            self.alfa.append(alfa_new)

    def local_search(self):
        """
        Local search phase of the algorithm
        """
        self.x_curr = self.best_x
        self.y_gen[self.epoch] = self.best_y
        self.alfa.append(self.alfa_best)
        while self.epoch < self.E:
            n = 0
            while n < self.N_local:
                dx = self.generate_dx(self.alfa[-1])
                self.epoch = self.epoch + 1
                if self.epoch > self.E:
                    break
                y_new = self.evaluate(self.x_curr + dx)
                positive = False
                if y_new < self.best_y:
                    self.best_y = y_new
                    self.best_x = self.x_curr + dx
                    self.y_gen[self.epoch] = self.y_gen[self.epoch-1]
                    n = n + 1
                    positive = True
                elif y_new < self.y_gen[self.epoch-1]:
                    self.y_gen[self.epoch] = y_new
                    self.x_curr = self.x_curr + dx
                    n = n + 1
                    positive = True
                if not positive:
                    y_new = self.evaluate(self.x_curr - dx)
                    if y_new < self.best_y:
                        self.best_y = y_new
                        self.best_x = self.x_curr - dx
                        self.y_gen[self.epoch] = self.y_gen[self.epoch-1]
                        n = n + 1
                    elif y_new < self.y_gen[self.epoch-1]:
                        self.y_gen[self.epoch] = y_new
                        self.x_curr = self.x_curr - dx
                        n = n + 1
                    else:
                        self.y_gen[self.epoch] = self.y_gen[self.epoch - 1]
                self.log({'epoch': self.epoch, 'n': n, 'alfa': self.alfa[-1], 'y_new': y_new, 'y_curr': self.y_gen[self.epoch-1], 'y_best': self.best_y, 'x_best': self.best_x})
            self.k = self.k + 1
            alfa_new = self.alfa[-1]*0.5
            self.alfa.append(alfa_new)
        raise StopCriterion('Exhausted maximum allowed number of epochs')

    def evaluate(self, x):
        """
        Single evaluation of the objective function
        :param x: point to be evaluated
        :return: corresponding objective function value
        """
        x = self.correction.correct(x)
        y = self.of.evaluate(x)
        self.neval += 1
        if y <= self.of.get_fstar():
            self.best_y = y
            self.best_x = x
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def search(self):
        """
        DRASET algorithm
        """
        try:
            self.x_curr = self.of.generate_point()
            self.y_gen[0] = self.of.evaluate(self.x_curr)
            self.best_y = self.y_gen[0]
            self.best_x = self.x_curr
            self.alfa_best = self.alfa[-1]
            self.general_search()
            self.local_search()
        except StopCriterion:
            return self.report_end()
        except:
            raise

    def report_end(self):
        """
        Returns report after heuristic has finished
        :return: dict with all necessary data
        """
        vysl = {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'epoch': self.epoch if self.best_y <= self.of.get_fstar() else np.inf,
            'neval': self.neval if self.best_y <= self.of.get_fstar() else np.inf,
            'alfa': self.alfa,
            'f_x': self.y_gen,
            'log_data': pd.DataFrame(self.log_data)
        }
        return vysl
