class ObjFun(object):

    """
    Generic objective function super-class
    """

    def __init__(self, fstar, a, b, name='abstract'):
        """
        Default initialization function that sets:
        :param fstar: f^* value to be reached (can be -inf)
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        self.fstar = fstar
        self.a = a
        self.b = b
        self.name = name

    def get_name(self):
        """
        Returns the name of the object function the object is representing.
        """
        return self.name

    def get_fstar(self):
        """
        Returns f^*
        :return: f^* value
        """
        return self.fstar

    def get_bounds(self):
        """
        Returns domain bounds
        :return: list with lower and upper domain bound
        """
        return [self.a, self.b]

    def generate_point(self):
        """
        Random point generator placeholder
        :return: random point from the domain
        """
        raise NotImplementedError("Objective function must implement its own random point generator")

    def get_neighborhood(self, x, d=1):
        """
        Solution neighborhood generating function placeholder
        :param x: point
        :param d: diameter
        :return: list of points in the neighborhood of the x
        """
        raise NotImplementedError("Objective function must implement its own neighborhood generator")

    def evaluate(self, x):
        """
        Objective function evaluating function placeholder
        :param x: point
        :return: objective function value
        """
        raise NotImplementedError("Objective function must implement its own evaluation")
