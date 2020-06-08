# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:19:54 2020

@author: Radovan
"""

from heur import StopCriterion, Heuristic
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


class GlowwormSwarm(Heuristic):

    def __init__(self, of, maxeval, n, rs, rho=None, gamma=None, beta=None, l0=None, s=None, eps=None):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        
        n       population
        rs      max sensor range
        rho     luciferin decay
        gamma   luciferin enhancement
        beta
        nt      number of neighbours
        l0      initial luciferin
        s       step
        """
        
        self.of = of
        self.maxeval = maxeval
        self.eps = eps or 1e-3 * self.of.get_fstar()
        
        self.neval = 0
        self.best_y = np.inf
        self.best_x = None

        assert n > 0, "Zero population"
        self.n = n
        assert rs > 0, "Zero sensor range"
        self.rs = rs
        self.rho = rho or 0.4
        assert self.rho > 0 and self.rho <= 1, "Decay out of range"
        self.gamma = gamma or 0.6
        self.beta = beta or 0.08
        self.l0 = l0 or 5
        self.s = s or 0.03
        
        self.log_data = []
        self.x = None
        self.f = np.zeros(self.n)
        
    
    def evaluate(self, x):
        """
        Single evaluation of the objective function
        :param x: point to be evaluated
        :return: corresponding objective function value
        """
#        f = np.zeros(self.n)
#        for i in range(self.n):
#            f[i] = self.of.evaluate(x[i, :])
#            self.neval += 1
        f = self.of.evaluate(x)
        self.neval += self.n
        y = np.min(f)
        if y < self.best_y:
            self.best_y = y
            self.best_x = x[np.argmin(f), :]
        if y <= self.of.get_fstar() + self.eps:
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval >= self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return f
    

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        self.reset()
        try:
            
            # initial values
            luc = np.full(self.n, self.l0)
            r = np.full(self.n, self.rs)
            # deploy  agents
            self.x = self.of.generate_point(self.n)
            while True:
                # luciferin update
                self.f = self.evaluate(self.x)
                luc = (1-self.rho) * luc + self.gamma * self.f
                # neighborhood
                distmat = cdist(self.x, self.x)
                nei = np.logical_and(np.logical_and(distmat > self.s*1e-5, distmat < np.tile(r, (self.n, 1)).T), luc > luc[np.newaxis, :].T)
                # probability
                prob = nei * luc
                prob = np.true_divide(prob, np.sum(prob, axis=1)[np.newaxis, :].T, out=np.zeros_like(prob), where=prob > 0)
                # move agent
                y = self.x.copy()
                for i in range(self.n):
                    if np.allclose(prob[i, :], 0): continue
                    j = np.random.choice(self.n, 1, True, prob[i, :])
                    y[i, :] = self.x[i, :] + self.s * (self.x[j, :]-self.x[i,:]) / distmat[i, j]
                self.x = self.correct(y)
                # range update
                neidens = np.sum(nei, axis=1) / (np.pi * self.rs**2)
                r = self.rs / (1 + self.beta * neidens)
                
                self.log({'x': self.x, 'f': self.f.copy(), 'r': r.copy(), 'luc': luc.copy(), 'neidens': neidens.copy()})
                
        except StopCriterion:
            for i in range(self.n):
                self.f[i] = self.of.evaluate(self.x[i, :])
            return self.report_end()
        except:
            raise

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)
    
    
    def report_end(self):
        """
        Returns report after heuristic has finished
        :return: dict with all necessary data
        """
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'x': self.x,
            'y': self.f,
            'neval': self.neval if self.best_y <= self.of.get_fstar() + self.eps else np.inf,
            'log_data': pd.DataFrame(self.log_data)
        }