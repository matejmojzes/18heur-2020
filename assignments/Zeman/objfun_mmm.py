# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:09:06 2020

@author: Radovan
"""

from objfun import ObjFun
import numpy as np


class MMM(ObjFun):

    def __init__(self, dim):
        self.dim = dim or 2        
        self.fstar = -4.97750248659 * self.dim
        self.a = -2 * np.ones(self.dim)
        self.b = 2 * np.ones(self.dim)

    def generate_point(self, num=1):
        return self.a + np.random.rand(num, self.dim) * (self.b - self.a)

    def evaluate(self, x):
        xx = x.reshape([-1, self.dim])    
        y = -(3*xx**2*np.cos(xx)+xx**2*np.cos(4*xx)-xx**2*np.cos(8*xx))+(12*np.cos(2)+4*np.cos(8)-4*np.cos(16))
        y = np.sum(y, axis=1)
        return y