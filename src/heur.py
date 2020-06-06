import numpy as np
import pandas as pd


class StopCriterion(Exception):

    """
    Dedicated exception class, to make stopping of the heuristic more transparent
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Heuristic:

    """
    Generic heuristic super-class
    """

    def __init__(self, of, maxeval):
        """
        Initialization function
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        """
        self.of = of
        self.name = 'Abstract'
        self.maxeval = maxeval
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0
        self.log_data = []

    def get_name(self):
        """
        Returns name of the heuristic this object is representing.
        """
        return self.name

    def get_specs(self):
        """
        Get heuristic specification placeholder.
        Should return string identifying the heuristic including its set parameter values.
        """
        raise NotImplementedError("Heuristic must implement its own search function")

    def evaluate(self, x):
        """
        Single evaluation of the objective function
        :param x: point to be evaluated
        :return: corresponding objective function value
        """
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x
        if y <= self.of.get_fstar():
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval >= self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def log(self, data):
        """
        Appends a row of logging data
        :param data: dict with logging row
        """
        self.log_data.append(data)

    def clear(self):
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0
        self.log_data = []

    def report_end(self):
        """
        Returns report after heuristic has finished
        :return: dict with all necessary data
        """
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.of.get_fstar() else np.inf,
            'log_data': pd.DataFrame(self.log_data)
        }

    def search(self):
        """
        Core searching function placeholder
        :return: end result report
        """
        raise NotImplementedError("Heuristic must implement its own search function")
