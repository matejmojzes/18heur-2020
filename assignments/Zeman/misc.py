# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:41:59 2020

@author: Radovan
"""

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

def experiment(of, heur, num_runs, params, text='', desc='', optdist=None, optthreshold=np.inf):
    results = []
    #for i in tqdm(range(num_runs), 'Testing {}'.format(params), ncols=800):
    for i in tqdm(range(num_runs), text, ncols=800):
        result = heur(of, **params).search()
        result['run'] = i
        if optdist:
            opt_x, opt_y = optima(result['x'].copy(), result['y'].copy(), optdist, optthreshold)
            result['opt_y'] = opt_y
            result['opt_x'] = opt_x
            result['numopt'] = np.sum(np.array(opt_y) < optthreshold)                                                                                                                                                   
        results.append({**result, **params, **desc})
    return pd.DataFrame(results, columns=[key for key in desc]+[key for key in params]+['run', 'best_x', 'best_y', 'neval']+(['opt_x', 'opt_y', 'numopt'] if optdist else []))

def optima(x, y, optdist=1, threshold=np.inf):
    opt_y = []
    opt_x = []
    while True:
        i = np.argmin(y)
        if y[i] >= threshold: break
        opt_y.append(y[i])
        opt_x.append(x[i, :])
        i = np.linalg.norm(x - x[i, :], axis=1) < optdist
        y[i] = np.inf
    return opt_x, opt_y

def rel(x):
    return len([n for n in x if n < np.inf])/len(x)

def mne(x):
    return np.mean([n for n in x if n < np.inf])

def feo(x):
    return mne(x)/rel(x)

def mcmult(k, M=4, runs=100000):
    x = np.random.multinomial(k, np.ones(M) / M, size=runs)
    return np.mean(np.all(x, axis=1))