# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:02:24 2020

@author: Ales
"""
#import external libraries
import numpy as np
import pandas as pd
import os

# import our modules
from hyperparameters import heuristic_methods
from heur_sg import ShootAndGo


def rel(x):
    return len([n for n in x if n < np.inf])/len(x)

def mne(x):
    return np.mean([n for n in x if n < np.inf])

def feo(x):
    return mne(x)/rel(x)

def pivot_table(df, index=['random_descent'], values=['feo']):
    stats = df.pivot_table(
        index = index,
        values = values,
        aggfunc = (feo)
    )["feo"]
    stats = stats.reset_index()
    return stats
    
def create_table(res, first_order_heur):
    res = round(res,2)
    width = res.shape[0]
    best_x = np.array(res["best_x"]).reshape(width, 1)
    try:
        length = best_x[0][0].size
    except:
        try:
            length = len(best_x[0][0])
        except:
            return res        
    rounded = np.zeros((width, length))
    for i in range(best_x.size):
        row = np.array(best_x[i])
        for j in range(len(row[0])):
            rounded[i,j] = '%s' % float('%.3g' % row[0][j])

    res = res.drop(["best_x"], axis=1)
    heur_dictionary = heuristic_methods[str(first_order_heur)]
    table = pd.DataFrame(rounded.tolist(), columns = heur_dictionary["pars"])
    table = table.join(res[["best_y"]])
    table = table.rename(columns={"best_y": "feo"})
    table.insert(loc=0, column='first_order_heur', value=first_order_heur)
    return table
    
def second_order_experiment(second_order_of, first_order_heur,
                    second_order_exp_num_runs, second_order_heur_maxeval,
                    second_order_heur_hmax, second_order_heur_random_descent):
    results = []
    print("==============Start of second order experiment==============")
    print("PARAMETERS OF {}:".format(first_order_heur),
          heuristic_methods[str(first_order_heur)]["pars"])
    for i in range(second_order_exp_num_runs):
        print("======{}-th run of second order experiment with {} evals======"\
              .format(i, second_order_heur_maxeval))
        result = ShootAndGo(second_order_of,maxeval =second_order_heur_maxeval,
                    hmax = second_order_heur_hmax,
                    random_descent=second_order_heur_random_descent).search()
        results.append(result)
        #print("RESULTS:", results)
    results = pd.DataFrame(results, columns=['best_x', 'best_y'])
    print("=========================FINAL TABLE=========================")
    table = create_table(results, first_order_heur)
    return table

def create_files(first_order_heur, mode="a", delete=False):
    """
     We are going to save results into files, we check if file exist and if
     not we create it. If user will not delete files and run second order
     objective function with first order heuristic, results will be appended
     to corresponding file. This can be wanted, if user want to make analysis
     on larger datafiles.
    """
    # ---------------------first order experiment results--------------------
    if not os.path.exists(os.path.join("..", "results",\
                                       "{}_first_order_experiment.txt".\
                                       format(first_order_heur))) or delete:
        with open(os.path.join('..', 'results',\
                               '{}_first_order_experiment.txt'.\
                               format(first_order_heur)), '{}'.\
                               format(mode)) as file:
            file.write("run,best_x,best_y,neval" + "\n")
    # ---------------------second order experiment results---------------------
    if not os.path.exists(os.path.join("..", "results",\
                                       "{}_second_order_experiment.txt".\
                                       format(first_order_heur))) or delete:
        header = str(heuristic_methods[first_order_heur]["pars"])
        header = header.replace("]", "").replace("[", "").replace(" ", "").\
                    replace("'", "")
        with open(os.path.join('..', 'results',\
                               '{}_second_order_experiment.txt'.\
                               format(first_order_heur)), '{}'.\
                               format(mode)) as file:
            file.write(header + ",feo" "\n")

def read_data(first_order_heur, experiment="second"):
    """
    Read data from saved file of results first or second order experiment runs.
    """
    if experiment == "second":
        data = pd.read_csv(os.path.join('..', 'results', 
                        '{}_second_order_experiment.txt'\
                        .format(first_order_heur)), sep=",")
    elif experiment == "first":
        data = pd.read_csv(os.path.join('..', 'results', 
                        '{}_first_order_experiment.txt'\
                        .format(first_order_heur)), sep=",")
    else:
        raise Exception("Wrong arguments!")
    return data

def delete_files(first_order_heur):
    """
    We delete all data excluding header in given files.
    """
    create_files(first_order_heur, mode="w", delete=True)

def give_object_name(obj_name, namespace):
    """
    Convert name of variable to string
    """
    name = [n for n in namespace if namespace[n] is obj_name]
    name = str(name)
    name = name.replace("]", "").replace("[", "").replace("'", "")
    return name
    
def save_object(obj_name, namespace, mode="w"):
    """
    Saving to results folder of content of variable to file of variable name.
    We want to sometimes save our results, because experiments are time
    consuming and we want to have that data, if we overwrite for example 
    in a local python scope.
    """
    name = [n for n in namespace if namespace[n] is obj_name]
    name = str(name)
    name = name.replace("]", "").replace("[", "").replace("'", "")
    with open(os.path.join('..', 'results', '{}'.format(name)), "{}"\
              .format(mode)) as file:
        if mode == "a":
            file.write(str(obj_name).split("\n",1)[1] + "\n")
        else:
            file.write(str(obj_name) + "\n")
            
def mean_and_variance(results_heur):
    print("====Average of hyperparameters of top performing runs:====")
    mean = results_heur.mean(axis=0).round(decimals=2)
    var = results_heur.var(axis=0).round(decimals=2)
    mean_and_var = pd.concat([mean, var], axis=1)
    mean_and_var.columns=["MEAN", "VARIANCE"]
    mean_and_var = mean_and_var.T
    mean_and_var.insert(0, "         ", "", True)
    print(mean_and_var)