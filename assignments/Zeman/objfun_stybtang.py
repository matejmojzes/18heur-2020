# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:09:06 2020

@author: Radovan
"""

from objfun import ObjFun
import numpy as np


class StyblinskiTang(ObjFun):

    def __init__(self, dim):
        self.dim = dim or 2        
        self.fstar = (-39.1661656-125) * self.dim
        self.a = -5 * np.ones(self.dim)
        self.b = 5 * np.ones(self.dim)

    def generate_point(self, num=1):
        return self.a + np.random.rand(num, self.dim) * (self.b - self.a)

    def evaluate(self, x):
        xx = x.reshape([-1, self.dim])    
        y = xx ** 4 - 16 * xx ** 2 + 5 * xx - 250
        y = np.sum(y, axis=1) / 2
        return y

def isStyblinskiTangOptimum(x, order=1, tol=1):
    j = np.array([-2.90351563, 2.74677734]).reshape(2, 1)
    d = np.abs(x-j)
    i = np.argmin(d, 0)
    return np.linalg.norm(d[i, :]) < tol and np.sum(i) <= order

def StyblinskiTangOptimum(dim):
    # nejlepší tři řady minim
    x = np.full(dim, -2.90351563)
    y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)
    xx = [x]
    yy = [y]
    for i in range(dim):
        for j in range(i, dim):
            x = np.full(dim, -2.90351563)
            x[i] = 2.74677734
            x[j] = 2.74677734
            y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)
            xx.append(x)
            yy.append(y)
    return xx, yy