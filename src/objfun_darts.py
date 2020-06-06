import math
import numpy as np
from objfun import ObjFun

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def cart_to_pol(x):
    alpha = np.arctan2(x[1], x[0])
    distance = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return [alpha, distance]


def pol_to_car(alpha, distance):
    x = distance * np.cos(alpha)
    y = distance * np.sin(alpha)
    return [np.round(x, 1), np.round(y, 1)]


class Dartboard(ObjFun):

    """
    Object function representing any kind of dartboard that can be defined by concentric circles and radial sectors.
    Default constructor values defines standard dartboard.
    """

    def __init__(self, sectors=None, sectors_angle=None, ring_params=None):
        """
        Initialization function
        :param sectors: list of scores attributable to radial sectors. Length of the list defines the number of them.
        :param sectors_angle: initial angular rotation of all sectors
        :param ring_params: defines concentric circles that modify the sector score. Each circle is defined by its
                            perimeter, sectors score multiplier and constant to add to the score. Expects list of lists.
        """
        if sectors is not None:
            self.sectors = sectors
        else:
            self.sectors = np.array([6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10])

        if ring_params is not None:
            self.ring_params = ring_params
        else:
            self.ring_params = np.array([
                [6.35, 0, 50],
                [15.9, 0, 25],
                [107, 1, 0],
                [115, 3, 0],
                [162, 1, 0],
                [170, 2, 0],
            ])

        if sectors_angle is not None:
            self.dartboard_angle = sectors_angle
        else:
            self.dartboard_angle = math.pi / len(self.sectors)

        self.sector_angle = 2 * math.pi / len(self.sectors)
        name = 'dartboard'
        a = np.array([-1 * self.ring_params[-1][0], -1 * self.ring_params[-1][0]])
        b = np.array([self.ring_params[-1][0], self.ring_params[-1][0]])
        super().__init__(-1 * self.count_max_score(), a, b, name)

    def evaluate(self, x):
        alpha, distance = cart_to_pol(x)
        if not isinstance(alpha, np.ndarray):
            alpha = np.array([alpha])
            distance = np.array([distance])

        index = alpha < 0
        alpha[index] = 2 * math.pi + alpha[index]
        index = (alpha + self.dartboard_angle) // self.sector_angle
        index = index.astype(int)
        index[index == 20] = 0

        score = self.sectors[index]

        score[distance > self.ring_params[-1][0]] = 0

        for i in range(len(self.ring_params) - 1, 0, -1):
            index = np.logical_and(self.ring_params[i][0] > distance, distance >= self.ring_params[i - 1][0])
            score[index] = self.ring_params[i][1] * score[index] + self.ring_params[i][2]
        index = distance < self.ring_params[0][0]
        score[index] = self.ring_params[0][1] * score[index] + self.ring_params[0][2]
        score = -1 * score
        return score

    def count_max_score(self):
        max_points = max(self.sectors)
        return max(max_points * self.ring_params[:, 1] + self.ring_params[:, 2])

    def generate_point(self):
        x, y = pol_to_car(np.random.uniform(0, 2 * math.pi, 1), np.random.uniform(0, self.a, 1))
        return np.array([x[0], y[0]])

    def get_neighborhood(self, x, d=2):
        epsilon = 1  # mm
        neighborhood = []
        searched_points = [list(x)]
        for round in range(d):
            new_searched_points = []
            for point in searched_points:
                for i in range(len(point)):
                    if point[i] > self.a[i]:
                        xx = point.copy()
                        xx[i] -= epsilon
                        if xx not in searched_points:
                            new_searched_points.append(xx)
                        if xx not in neighborhood:
                            neighborhood.append(xx)
                    if point[i] < self.b[i]:
                        xx = point.copy()
                        xx[i] += epsilon
                        if xx not in searched_points:
                            new_searched_points.append(xx)
                        if xx not in neighborhood:
                            neighborhood.append(xx)
            searched_points = new_searched_points
        return neighborhood


class DartsAvgScore(ObjFun):

    """
    Object encapsulating forwarded object function. It can serve as an object function too, but returns average score
    of defined number of evaluations of encapsulated of. The forwarded point for evaluation serves as a mean value of
    normal distribution. Effectively it just blurs encapsulated of.
    """

    def __init__(self, variance, iterations=None, dartboard=None):
        """
        Initialization function
        :param variance: variance or covariance matrix for normal distribution defining throws accuracy
        :param iterations: number of random throws from which the average score is counted
        :param dartboard: encapsulated object function representing the dartboard itself
        """
        if isinstance(variance, list):
            if isinstance(variance[0], list):
                self.covariance = variance
            else:
                self.covariance = [[variance[0], 0], [0, variance[1]]]
        else:
            self.covariance = [[variance, 0], [0, variance]]

        if iterations is None:
            self.iterations = 30000
        else:
            self.iterations = iterations

        if dartboard is None:
            self.dartboard = Dartboard()
        else:
            self.dartboard = dartboard

        self.sectors = self.dartboard.sectors
        self.ring_params = self.dartboard.ring_params
        self.dartboard_angle = self.dartboard.dartboard_angle
        self.sector_angle = self.dartboard.sector_angle
        name = 'dartsscore'
        super().__init__(self.dartboard.fstar, self.dartboard.a, self.dartboard.b, name)

        self.cache = np.full([int((self.b[0] - self.a[0]) * 10), int((self.b[1] - self.a[1]) * 10)], np.nan,
                             dtype=np.float)

    def generate_point(self):
        return self.dartboard.generate_point()

    def get_neighborhood(self, x, d):
        return self.dartboard.get_neighborhood(x, d)

    def evaluate(self, x):
        """
        :param x: 2D point
        :return: objective function value
        """
        if isinstance(x[0], np.ndarray):
            x1 = x[0]
            x2 = x[1]
        else:
            x1 = np.array([x[0]])
            x2 = np.array([x[1]])

        m, n = self.cache.shape
        avg_score = np.full(x1.shape, 0, dtype=float)
        for i in range(len(x1)):
            x1_index = int((x1[i] + self.b[0]) * 10)
            x2_index = int((x2[i] + self.b[1]) * 10)
            if m > x1_index and n > x2_index and x1_index >= 0 and x2_index >= 0:
                if np.isnan(self.cache[x1_index, x2_index]):
                    avg_score[i] = np.average(self.dartboard.evaluate(
                        np.random.multivariate_normal([x1[i], x2[i]], self.covariance, self.iterations).T))
                    self.cache[x1_index, x2_index] = avg_score[i]
                else:
                    avg_score[i] = self.cache[x1_index, x2_index]
            else:
                avg_score[i] = 0
        return avg_score


class DartsPlotter(object):

    """
    Plots forwarded dartboard object function. Other of should not be forwarded.
    """

    def __init__(self):
        self.epsilon = 0.000001
        fig = plt.figure()
        self.ax = Axes3D(fig)

    def plot_dartboard(self, dartboard):
        x = self.circuit_points_meshgrid(0.01, 1, 230)

        score = dartboard.evaluate([np.ravel(x[0]), np.ravel(x[1])])

        score = -1 * score.reshape(x[0].shape)

        self.plot_flat_dartboard(dartboard)

        self.ax.plot_surface(x[0], x[1], score, linewidth=0, alpha=0.9, vmin=0,
                             rstride=10, cstride=10, antialiased=False, cmap='pink', shade=True)
        self.set_axis()

    def scatter_points(self, dartboard, x):
        self.plot_flat_dartboard(dartboard)
        self.ax.scatter(x[0], x[1], -1 * dartboard.evaluate(x), cmap='coolwarm')
        self.set_axis()

    def plot_points(self, dartboard, x, y):
        self.plot_flat_dartboard(dartboard)
        z = []
        for array in y:
            z.append(-1*array[0])

        for i in range(1, len(x[0])):
            self.ax.plot([x[0][i - 1], x[0][i]], [x[1][i - 1], x[1][i]], [z[i - 1], z[i]],
                         c=[180 / 255, 180 / 255, 180 / 255], linewidth=1)
        self.ax.scatter(x[0], x[1], z, cmap='coolwarm')
        self.set_axis()

    def show_numbers(self, dartboard):
        number_angles = np.arange(0, 2 * math.pi, dartboard.sector_angle)
        numbers_distance = 0 * number_angles + dartboard.ring_params[-1][0] * 1.2
        x = numbers_distance * np.cos(number_angles)
        y = numbers_distance * np.sin(number_angles)
        for i in range(len(number_angles)):
            self.ax.text(x[i], y[i], 0, dartboard.sectors[i], horizontalalignment='center', fontsize=12,
                         color=[180 / 255, 180 / 255, 180 / 255])

    def set_axis(self):
        self.ax.set_xlabel('Osa: x [mm]')
        self.ax.set_ylabel('Osa: y [mm]')
        self.ax.set_zlabel('Osa: Score')
        self.ax.set_xlim3d(-250, 250)
        self.ax.set_ylim3d(-250, 250)
        self.ax.set_zlim3d(0, 60)

        self.ax.view_init(elev=85, azim=-90)

    def circuit_points_meshgrid(self, step_angle, step_dist, radius):
        f = np.arange(0, 2 * math.pi, step_angle)
        f = np.append(f, 2 * math.pi - self.epsilon)
        [alpha, distance] = np.meshgrid(f, np.arange(0, radius, step_dist))
        # alpha = alpha.flatten()
        # distance = distance.flatten()

        x = np.multiply(distance, np.cos(alpha))
        y = np.multiply(distance, np.sin(alpha))
        return [x, y]

    def draw_ellipse(self, x0, y0, xR, yR):
        t = np.arange(-1 * math.pi, math.pi, 0.01)
        x = x0 + xR * np.cos(t)
        y = y0 + yR * np.sin(t)
        self.ax.plot(x, y, 0, c='black', linewidth=0.5)

    def plot_flat_dartboard(self, dartboard):

        for i in range(0, len(dartboard.ring_params)):
            self.draw_ellipse(0, 0, dartboard.ring_params[i][0], dartboard.ring_params[i][0])

        lines_angle = np.arange(dartboard.dartboard_angle, 2 * math.pi, dartboard.sector_angle)
        lines_start = 0 * lines_angle + dartboard.ring_params[1][0]
        lines_end = 0 * lines_angle + dartboard.ring_params[-1][0] * 1.3

        start_x, start_y = pol_to_car(lines_angle, lines_start)
        end_x, end_y = pol_to_car(lines_angle, lines_end)

        for i in range(len(start_x)):
            self.ax.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], c='black', linewidth=0.5)
        self.show_numbers(dartboard)
