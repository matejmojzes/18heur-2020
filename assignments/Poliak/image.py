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
        xs = np.ones(numTriangles * 3, dtype=np.uint8) * self.width
        ys = np.ones(numTriangles * 3, dtype=np.uint8) * self.height
        points = np.empty((xs.size + ys.size,),dtype=np.uint8)
        points[0::2] = xs
        points[1::2] = ys
        colors = np.ones(numTriangles * 4, dtype=np.uint8) * 255
        super().__init__(fstar=fstar, a=np.zeros(self.dim, dtype=np.uint8), b=np.concatenate((points, colors)))


    def generate_point(self):
        xs = rg.integers(low=0, high=self.height, size=self.numTriangles * 3, dtype=np.uint8)
        ys = rg.integers(low=0, high=self.height, size=self.numTriangles * 3, dtype=np.uint8)
        points = np.empty((xs.size + ys.size,), dtype=np.uint8)
        points[0::2] = xs
        points[1::2] = ys
        colors = rg.integers(low=0, high=255,size=4*self.numTriangles, endpoint=True, dtype=np.uint8)
        return np.concatenate((points, colors))
    
    def evaluate(self, x):
        im = self.interpret(x)
        arr = np.array(im).reshape((self.cells,)).astype(np.int16)
        return np.linalg.norm(self.source - arr)

    def interpret(self, x) -> Image:
        im = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(im, "RGBA")
        points = x[:self.numTriangles * 2 * 3].reshape((self.numTriangles, 6))
        colors = x[self.numTriangles * 2 * 3:].reshape((self.numTriangles, 4))
        for t in range(self.numTriangles):
            draw.polygon(list(points[t]), tuple(colors[t]))
        return im


def save_jpg(name: str, im: Image):
    im.save(f"{name}.jpg", "JPEG",subsampling=0, quality=100)