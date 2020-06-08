# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:05:05 2020

@author: Ales
"""
from heur_aux import HyperparametersCorrection
from heur_sg import ShootAndGo
from heur_fsa import FastSimulatedAnnealing
from heur_go import GeneticOptimization
from heur_de import DifferentialEvolution

from objfun import ObjFun

from hyperparameters import heuristic_methods
from stats import feo

import pandas as pd
import numpy as np
import time
import copy
import math
import os

class TuneHyperparameters(ObjFun):
    """
    First I am only optimizing shoot and go hyperparameters.
    So this is second order objective function.
    """
    
    def __init__(self, heur, of, time):
        """
        Initialization function
        :param heur_dictionary: choose right dictinary, which contains first
        order heuristics and bounds for experiment
        :param heur: choose first order heuristic method from dictionary,
        options: "SG", "FSA", "GO", "DE"
        :param a: lower bound for first order experiment
        :param b: upper bound for first order experiment
        :param of: first order objective function
        :param fstar: what is the best possible value of feo criterion?
        :param Corr: correction method for first order hyperparameters
        :param time_of_experiment: time of first order experiment
        """
        self.heur_name = str(heur)
        self.heur_dictionary = heuristic_methods[self.heur_name]
        self.heur = self.heur_dictionary["heur"]
        self.a = np.array(self.heur_dictionary["a"])
        self.b = np.array(self.heur_dictionary["b"])
        self.of = of # first order objective function
        self.fstar = 1 # f* of feoquist criterion
        self.Corr = HyperparametersCorrection(self.of,
                                              self.heur_dictionary["discrete"])
        # how many seconds we have on each first order experiment
        self.time_of_experiment = time
        self.print = True # printing evaluation results
        self.save = True # saving data
        self.path_to_save = os.path.join('results', \
                           '{}_parameters.txt'.format(self.heur_name))
               
    def first_order_experiment(self, *hyperparameters):
        """
        We run first order experiment with first order hyperparameter setup so
        many times which our processor can run in a given time.
        Because different first order heuristic methods have different number
        of first order hyperparameters, we use *.
        In self.heur there is first order heuristic function.
        """
        start = time.time()
        results = []
        while True:
            # dict with results of one run
            result = self.heur(*hyperparameters[1:]).search()
            results.append(result)
            # if we exceeded the time limit for experiment
            if  (time.time() - start) > self.time_of_experiment:
                break        
        return pd.DataFrame(results, columns=['best_x', 'best_y', 'neval'])
      
    def generate_point(self):
        """
        According to heuristic function, we randomly generate first order
        hyperparameter setup.
        """
        if self.heur == ShootAndGo:
            maxeval = int(np.floor(np.random.triangular(self.a[0], self.a[0],
                                                    self.b[0]+1)))
            hmax = int(np.floor(np.random.exponential(3)))
            random_descent = np.random.randint(2)
            return [self.heur, self.of, maxeval, hmax, random_descent]
        
        elif self.heur == FastSimulatedAnnealing:
            T0 = 10**np.random.uniform(-10,3)
            maxeval = int(np.floor(np.random.triangular(self.a[1], self.a[1],
                                                    self.b[1]+1)))
            n0 = np.random.triangular(self.a[2], self.a[2], self.b[2])
            alpha = np.random.triangular(self.a[3], self.a[3], self.b[3])
            return [self.heur, self.of, T0, maxeval, n0, alpha]
        
        elif self.heur == GeneticOptimization:
            maxeval = int(np.floor(np.random.triangular(self.a[0], self.a[0],
                                                    self.b[0]+1)))
            N = int(np.floor(np.random.triangular(self.a[1], self.a[1],
                                                    self.b[1]+1)))
            M = int(np.floor(np.random.triangular(self.a[2], self.a[2],
                                                    self.b[2]+1)))
            Tsel1 = np.random.triangular(self.a[3], self.a[3], self.b[3])
            Tsel2 = np.random.triangular(self.a[4], self.a[4], self.b[4])
            return [self.heur, self.of, maxeval, N, M, Tsel1, Tsel2]
        elif self.heur == DifferentialEvolution:
            maxeval = int(np.floor(np.random.triangular(self.a[0], self.a[0],
                                                    self.b[0]+1)))
            N = int(np.floor(np.random.triangular(self.a[1], self.a[1],
                                                    self.b[1]+1)))
            CR = np.random.triangular(self.a[2], self.a[2], self.b[2])
            F = np.random.triangular(self.a[3], self.a[3], self.b[3])
            return [self.heur, self.of, maxeval, N, CR, F]
        
    def evaluate(self, hyperparameters):
        """
        We run first order experiment and our score is feo criterion from that
        experiment.
        """
        # CHECKING DIMENSIONS-------------------------------------------------
        if "numpy" in str(type(hyperparameters[0])): # there is no of at first
            hyperparameters = list(hyperparameters)
            hyperparameters.insert(0, self.of)
            hyperparameters.insert(0, self.heur)
            
        # EVALUATING----------------------------------------------------------
        table = self.first_order_experiment(*hyperparameters)
        
        # ROUNDING------------------------------------------------------------
        feo_value = feo(table["neval"]) if not math.isnan(feo(table["neval"]))\
            else np.inf
        feo_value = round(feo_value, 2)
        
        # PRINTING------------------------------------------------------------
        if self.print:
            rounded_hyps = []
            for i in range(len(hyperparameters[2:])):
                rounded_hyps.append(float("{:.3f}".\
                                          format(hyperparameters[i+2])))
            print("Parameters_of_{}: {}, feo: {}".format(self.heur_name,
                  rounded_hyps, feo_value))
         
        # SAVING--------------------------------------------------------------
        if self.save: 
            table.to_csv(os.path.join('..', 'results', \
                    '{}_first_order_experiment.txt'.format(self.heur_name)),\
                    mode='a', header=False)                  
            with open(os.path.join('..', 'results', \
                           '{}_second_order_experiment.txt'.\
                           format(self.heur_name)), 'a') as file:
                file.write('{}, {}'.format(str(rounded_hyps).replace("]", "").\
                            replace("[", ""), feo_value) + "\n")
        return feo_value

    def get_neighborhood(self, *hyperparameters):
        """
        For our first order experiment we generate neighbour points and we will
        make sure, ath the end, that our neighbours are not out of bounds.
        """
        # Different heuristics returns different dimensions, so that is the
        # reason why I am using following if statements.
        if "heur" in str(hyperparameters):
            hyperparameters = np.array(hyperparameters[0][2:])
        else:
            hyperparameters = np.array(hyperparameters[0])
        # Put hyperparameters inside bounds, if they are out.
        hyperparameters = self.Corr.correct(hyperparameters, self.a, self.b)
        
        discrete = self.heur_dictionary["discrete"]
        neighbours = []
        for i in range(len(self.a)):
            if self.a[i] == self.b[i]:
                pass 
            # if hyperparameter is discrete
            elif discrete[i]:
                width_of_triangular_dist =np.ceil(np.minimum(hyperparameters[i] 
                    - self.a[i], self.b[i] - hyperparameters[i]) / 2)

                if width_of_triangular_dist == 0: # if we are on the border
                    width_of_triangular_dist = self.b[i] - self.a[i]
                # we want left and right neighbour
                shift = np.ceil(np.random.triangular(0, 0, 
                                                     width_of_triangular_dist))
                # left neighbour
                neighbour1 = copy.copy(hyperparameters) # deep copy
                neighbour1[i] = hyperparameters[i] - shift
                # right neighbour
                neighbour2 = copy.copy(hyperparameters) # deep copy
                neighbour2[i] = hyperparameters[i] + shift
                neighbours.append(neighbour1)
                neighbours.append(neighbour2)
            # if hyperparameter is continuous
            elif not discrete[i]:
                width_of_triangular_dist = np.minimum(hyperparameters[i] 
                    - self.a[i], self.b[i] - hyperparameters[i]) / 2
                if width_of_triangular_dist == 0: # if we are on the border
                    width_of_triangular_dist = self.b[i] - self.a[i]
                # we want left and right neighbour
                shift = np.random.triangular(0, 0, width_of_triangular_dist)
                # left neighbour
                neighbour1 = copy.copy(hyperparameters) # deep copy
                neighbour1[i] = hyperparameters[i] - shift
                # right neighbour
                neighbour2 = copy.copy(hyperparameters) # deep copy
                neighbour2[i] = hyperparameters[i] + shift
                neighbours.append(neighbour1)
                neighbours.append(neighbour2)
        # we get rid of out of bounds neighbours
        neighbours_inside_bounds = []
        for i, n in enumerate(neighbours):
            if self.is_in_bounds(n):
                neighbours_inside_bounds.append(n)
        return neighbours_inside_bounds
             

        
        

    
    