# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:47:35 2020

@author: Ales
"""
from heur_sg import ShootAndGo
from heur_fsa import FastSimulatedAnnealing
from heur_go import GeneticOptimization
from heur_de import DifferentialEvolution

"""
:param a: lower bound of first order experiment hyperparameters
:param b: upper bound of first order experiment hyperparameters
:param discrete: which of the first order experiment hyperparameters are
dicrete (1), which are continuous (0)
:param heur: first order heuristic method
"""
shoot_and_go = {
        "pars": ["maxeval", "hmax", "random_descent"],
        "a": [100, 0, 0],
        "b": [1000, 30, 1],
        "discrete": [1, 1, 1],
        "heur": ShootAndGo
        }

fast_simulated_annealing = {
        "pars": ["T0", "maxeval", "n0", "alpha"],
        "a": [1e-10, 100, 0.1, 0.5],
        "b": [1000, 1000, 3, 4],
        "discrete": [0, 1, 0, 0],
        "heur": FastSimulatedAnnealing,
        }

genetic_optimization = {
        "pars": ["maxeval", "N", "M", "Tsel1", "Tsel2"],
        "a": [100, 4, 10, 0.03, 0.3],
        "b": [1000, 8, 20, 0.3, 2],
        "discrete": [1, 1, 1, 0, 0],
        "heur": GeneticOptimization
        }

differential_evolution = {
        "pars": ["maxeval", "N", "CR", "F"],
        "a": [100, 4, 0.1, 0],
        "b": [1000, 20, 0.9, 2],
        "discrete": [1, 1, 0, 0],
        "heur": DifferentialEvolution
        }

"""
We are choosing which first order heuristic method (which of above dictionary)
we will choose
"""
heuristic_methods = { 
        "SG": shoot_and_go,
        "FSA": fast_simulated_annealing,
        "GO": genetic_optimization,
        "DE": differential_evolution
        }
