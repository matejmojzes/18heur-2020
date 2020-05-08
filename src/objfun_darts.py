import math
import numpy as np
import matplotlib
from objfun import ObjFun
from matplotlib.colors import LightSource

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm





def cart_to_pol(x):
    alpha = np.arctan2(x[1], x[0])
    distance = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return [alpha, distance]


def pol_to_car(alpha, distance):
    x = distance * np.cos(alpha)
    y = distance * np.sin(alpha)
    return [x, y]


class Dartboard(ObjFun):

    def __init__(self, sectors=None, sectors_angle=None, ring_params=None):

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

        super().__init__(-1*self.count_max_score(), [-1*self.ring_params[-1][0], -1*self.ring_params[-1][0]], [1*self.ring_params[-1][0], 1*self.ring_params[-1][0]])

    def evaluate(self, x):

        alpha, distance = cart_to_pol(x)
        index = alpha < 0
        alpha[index] = 2 * math.pi + alpha[index]
        index = (alpha + self.dartboard_angle) // self.sector_angle
        index = index.astype(int)
        index[index == 20] = 0

        score = self.sectors[index]

        score[distance > self.ring_params[-1][0]] = 0

        for i in range(len(self.ring_params)-1, 0, -1):
            index = np.logical_and(self.ring_params[i][0] > distance, distance >= self.ring_params[i - 1][0])
            score[index] = self.ring_params[i][1] * score[index] + self.ring_params[i][2]
        index = distance < self.ring_params[0][0]
        score[index] = self.ring_params[0][1] * score[index] + self.ring_params[0][2]
        score = -1*score
        return score

    def count_max_score(self):
        max_points = max(self.sectors)
        return max(max_points * self.ring_params[:, 1] + self.ring_params[:, 2])

    def generate_point(self):
        return pol_to_car(np.random.uniform(0, 2*math.pi, 1), np.random.uniform(0, self.a, 1))

    def get_neighborhood(self, x, d=1):
        epsilon = 0.01
        nd = []
        for i in range(len(x)):
            if x[i][0] > self.a[i]:
                xx = x.copy()
                xx[i][0] -= 0.01
                nd.append(xx)
            if x[i][0] < self.b[i]:
                xx = x.copy()
                xx[i][0] += 0.01
                nd.append(xx)
        return nd


class DartsAvgScore(ObjFun):

    """
    Generic objective function super-class
    """

    def __init__(self, variance, iterations=None, dartboard=None):
        """
        Default initialization function that sets:
        :param fstar: f^* value to be reached (can be -inf)
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        if isinstance(variance, list):
            if isinstance(variance[0], list):
                self.covariance = variance
            else:
                self.covariance = [[variance[0], 0], [0, variance[1]]]
        else:
            self.covariance = [[variance, 0], [0, variance]]

        if iterations is None:
            self.iterations = 100
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

        super().__init__(self.dartboard.fstar, self.dartboard.a, self.dartboard.b)

    def generate_point(self):
        """
        Random point generator placeholder
        :return: random point from the domain
        """
        return self.dartboard.generate_point()

    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function placeholder
        :param x: point
        :return: list of points in the neighborhood of the x
        """
        return self.dartboard.get_neighborhood(x)

    def evaluate(self, x):
        """
        Objective function evaluating function placeholder
        :param x: point
        :return: objective function value
        """
        avg_score = 0*x[0]
        for i in range(len(x[0])):
            avg_score[i] = np.average(self.dartboard.evaluate(np.random.multivariate_normal([x[0][i], x[1][i]], self.covariance, self.iterations).T))
        return avg_score


class DartsPlotter(object):

    def __init__(self):
        self.epsilon = 0.000001
        fig = plt.figure()
        self.ax = Axes3D(fig)

    def plot_dartboard(self, dartboard):
        x = self.circuit_points_meshgrid(0.01, 1, 230)

        score = dartboard.evaluate([np.ravel(x[0]), np.ravel(x[1])])

        score = -1*score.reshape(x[0].shape)

        self.plot_flat_dartboard(dartboard)

        self.ax.plot_surface(x[0], x[1], score, linewidth=0, alpha=0.9, vmin=0,
                        rstride=10, cstride=10, antialiased=False, cmap='pink', shade=True)
        self.set_axis()

    def scatter_points(self, dartboard, x):
        self.plot_flat_dartboard(dartboard)
        self.ax.scatter(x[0], x[1], -1*dartboard.evaluate(x), cmap='coolwarm')
        self.set_axis()

    def plot_points(self, dartboard, x):
        self.plot_flat_dartboard(dartboard)
        z = -1*dartboard.evaluate(x)

        for i in range(1, len(x[0])):
            self.ax.plot([x[0][i-1], x[0][i]], [x[1][i-1], x[1][i]], [z[i-1], z[i]], c=[180 / 255, 180 / 255, 180 / 255], linewidth=1)
        self.ax.scatter(x[0], x[1], z, cmap='coolwarm')
        self.set_axis()

    def show_numbers(self, dartboard):
        number_angles = np.arange(0, 2 * math.pi, dartboard.sector_angle)
        numbers_distance = 0*number_angles + dartboard.ring_params[-1][0] * 1.2
        x = numbers_distance * np.cos(number_angles)
        y = numbers_distance * np.sin(number_angles)
        for i in range(len(number_angles)):
            self.ax.text(x[i], y[i], 0, dartboard.sectors[i], horizontalalignment='center', fontsize=12, color=[180 / 255, 180 / 255, 180 / 255])

    def set_axis(self):
        self.ax.set_xlabel('Osa: x [mm]')
        self.ax.set_ylabel('Osa: y [mm]')
        self.ax.set_zlabel('Osa: Score')
        self.ax.set_xlim3d(-250, 250)
        self.ax.set_ylim3d(-250, 250)
        self.ax.set_zlim3d(0, 60)

        self.ax.view_init(elev=85, azim=-90)

    def circuit_points_meshgrid(self, step_angle, step_dist, radius):
        f = np.arange(0, 2*math.pi, step_angle)
        f = np.append(f, 2 * math.pi - self.epsilon)
        [alpha, distance] = np.meshgrid(f, np.arange(0, radius, step_dist))
        # alpha = alpha.flatten()
        # distance = distance.flatten()

        x = np.multiply(distance, np.cos(alpha))
        y = np.multiply(distance, np.sin(alpha))
        return [x, y]

    def draw_ellipse(self, x0, y0, xR, yR):
        t = np.arange(-1*math.pi, math.pi, 0.01)
        x = x0 + xR * np.cos(t)
        y = y0 + yR * np.sin(t)
        self.ax.plot(x, y, 0, c='black', linewidth=0.5)

    def plot_flat_dartboard(self, dartboard):

        for i in range(0, len(dartboard.ring_params)):
            self.draw_ellipse(0, 0, dartboard.ring_params[i][0],  dartboard.ring_params[i][0])

        lines_angle = np.arange(dartboard.dartboard_angle, 2 * math.pi, dartboard.sector_angle)
        lines_start = 0*lines_angle + dartboard.ring_params[1][0]
        lines_end = 0*lines_angle + dartboard.ring_params[-1][0] * 1.3

        start_x, start_y = pol_to_car(lines_angle, lines_start)
        end_x, end_y = pol_to_car(lines_angle, lines_end)

        for i in range(len(start_x)):
            self.ax.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], c='black', linewidth=0.5)
        self.show_numbers(dartboard)


