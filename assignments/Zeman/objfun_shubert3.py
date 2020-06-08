# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:21:49 2020

@author: Radovan
"""

from objfun import ObjFun
import numpy as np


class Shubert3(ObjFun):

    def __init__(self, dim, range='large'):
        self.dim = dim or 2        
        self.fstar = (-14.83794985-15) * self.dim
        if range == 'large': d = 10
        else: d = 2
        self.a = -d * np.ones(self.dim)
        self.b = d * np.ones(self.dim)

    def generate_point(self, num=1):
        return self.a + np.random.rand(num, self.dim) * (self.b - self.a)

    def evaluate(self, x):
        xx = x.reshape([-1, self.dim])    
        y = np.sin(2*xx+1) + 2*np.sin(3*xx+2) + 3*np.sin(4*xx+3) + 4*np.sin(5*xx+4) + 5*np.sin(6*xx+5)
        y = np.sum(y, axis=1)
        return y - 15 * self.dim

def isShubert3optimum(x, tol=1):
    j = np.array([5.16906738, -1.1140625, -7.39725342]).reshape(3, 1)
    return np.linalg.norm(np.min(np.abs(x-j), 0)) < tol