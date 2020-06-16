from image import TriangleDraw
from PIL import Image, ImageDraw
import numpy as np

im = Image.new("RGB", (100,100))
# draw = ImageDraw.Draw(im, "RGBA")
# draw.polygon([47, 61, 41, 78, 91, 97], (245, 181, 0, 118))
td = TriangleDraw(im, 1)
p = td.generate_point()
p
td.evaluate(p)


ar = np.array(im)


[198,  35, 227,   5,  28, 125]

