class ObjFun(object):

    """
    Generic objective function super-class
    """

    def __init__(self, fstar, a, b):
        """
        Default initialization function that sets:
        :param fstar: f^* value to be reached (can be -inf)
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        self.fstar = fstar
        self.a = a
        self.b = b

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

    def is_in_bounds(self, x):
        """
        Find out, if vector x is bounded by lower (a) and upper (b) bounds.
        """
        for i in range(len(self.a)):
            if not self.a[i] <= x[i] <= self.b[i]:
                return False
        return True

    def generate_point(self):
        """
        Random point generator placeholder
        :return: random point from the domain
        """
        raise NotImplementedError("Objective function must implement its own"
                                  "random point generator")

    def get_neighborhood(self, x):
        """
        Solution neighborhood generating function placeholder
        :param x: point
        :return: list of points in the neighborhood of the x
        """
        raise NotImplementedError("Objective function must implement its own"
                                  "neighborhood generator")

    def evaluate(self, x):
        """
        Objective function evaluating function placeholder
        :param x: point
        :return: objective function value
        """
        raise NotImplementedError("Objective function must implement its own"
                                  "evaluation")
                
        
