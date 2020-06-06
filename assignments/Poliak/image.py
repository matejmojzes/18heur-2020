import sys
sys.path.append("../../src")

from PIL import Image, ImageDraw
import numpy as np
from objfun import ObjFun
rg = np.random.default_rng(20)


class TriangleDraw(ObjFun):

    def __init__(self, im: Image, numTriangles: int, fstar: int = 0):
        self.im = im
        self.width = im.width
        self.height = im.height
        self.cells = im.width*im.height*3
        self.source = np.array(im).reshape((self.cells,)).astype(np.int16)
        self.numTriangles = numTriangles
        self.dim = numTriangles * 10
        xs = np.ones(numTriangles * 3, dtype=int) * self.width
        ys = np.ones(numTriangles * 3, dtype=int) * self.height
        points = np.empty((xs.size + ys.size,),dtype=int)
        points[0::2] = xs
        points[1::2] = ys
        colors = np.ones(numTriangles * 4, dtype=int) * 255
        super().__init__(fstar=fstar, a=np.zeros(self.dim, dtype=int), b=np.concatenate((points, colors)))


    def generate_point(self, triangles=None):
        if triangles is None:
            triangles = self.numTriangles
        xs = rg.integers(low=0, high=self.height, size=triangles * 3, dtype=int)
        ys = rg.integers(low=0, high=self.height, size=triangles * 3, dtype=int)
        points = np.empty((xs.size + ys.size,), dtype=int)
        points[0::2] = xs
        points[1::2] = ys
        colors = rg.integers(low=0, high=255,size=4*triangles, endpoint=True, dtype=int)
        return np.concatenate((points, colors))
    
    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            ds = 0
            while ds < d and x[i] - ds >= self.a[i]:
                xl = x.copy()
                xl[i] -= ds
                nd.append(xl)
                ds += 1
            # x-upper
            ds = 0
            while ds < d and x[i] + ds <= self.b[i]:
                xu = x.copy()
                xu[i] += ds
                nd.append(xu)
                ds += 1

        return nd
    
    def evaluate(self, x):
        im = self.interpret(x)
        arr = np.array(im).reshape((self.cells,)).astype(np.int16)
        return np.linalg.norm(self.source - arr)

    def interpret(self, x) -> Image:
        im = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(im, "RGBA")
        (points, colors) = self. separate(x)
        for t in range(self.numTriangles):
            draw.polygon(list(points[t]), tuple(colors[t]))
        return im

    def separate(self, x):
        points = x[:self.numTriangles * 2 * 3].reshape((self.numTriangles, 6))
        colors = x[self.numTriangles * 2 * 3:].reshape((self.numTriangles, 4))
        return (points.copy(), colors.copy())

    def combine(self, points, colors):
        res = np.empty((10* self.numTriangles,), dtype=int)
        res[: self.numTriangles * 6] = points.reshape((6* self.numTriangles,))
        res[self.numTriangles * 6:] = colors.reshape((4* self.numTriangles,))
        return res


class TriangleMutation:

    def __init__(self, of):
        self.of = of
        self.triangles = of.numTriangles

    def mutate(self, x):
        t_i = rg.integers(low=0, high=self.triangles, size=1, dtype=int)[0]
        subs = self.of.generate_point(1)
        (p, c) = self.of.separate(x)
        p[t_i] =  subs[:6]
        c[t_i] = subs[6:]
        return self.of.combine(p,c)


class TriangleCrossover:

    def __init__(self, of):
        self.of = of

    def crossover(self, x, y):
        t_is = rg.integers(low=0, high=2, size=self.of.numTriangles, dtype=int)
        (p1,c1) = self.of.separate(x)
        (p2,c2) = self.of.separate(y)
        ps = [p1,p2]
        cs = [c1,c2]
        p_res = np.array([ps[ti][i] for i, ti in enumerate(t_is)])
        c_res = np.array([cs[ti][i] for i, ti in enumerate(t_is)])
        return self.of.combine(p_res, c_res)


def save_jpg(name: str, im: Image):
    im.save(f"{name}.jpg", "JPEG",subsampling=0, quality=100)