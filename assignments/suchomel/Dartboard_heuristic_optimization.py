
# coding: utf-8

# # Dartboard as an Object Function
# ### Seminar work on Heuristic Optimization
# #### Author: Ales Suchomel
# #### Year: 2020

# #### WARNING! Re-running this notebook could take few hours...

# ### Introduction
# 
# This paper builds on [MMC seminar work (Czech)](files/sources/ales_suchomel_mmc_zapoctova_prace.pdf) that focused on problem definition, pseudo-random data generation and Monte-Carlo integration. Most of the theory, including domain explanation and derivation of the object function, is also already processed there with only one exception, the optimization itself. 
# 
# An interesting particle-flock based heuristic was used for that purpose, but its performance was not tested at all and it was used just as a black-box code implemented within Ondrej Panek's bachelor thesis. 
# 
# Therefore, the purpose of this paper is to complete the previous seminar work by finding the right heuristic for the optimization.

# ### Goals
# The formal goal of this paper is to find a point to aim the dart at that ensures the highest expected score from a single throw. Thanks to an interesting structure of the dartboard, the optimal point is not always in the middle of the target but rather depends on the player's skills. The final output should be a graph of development of the best strategy depending on player's throw variance.
# 
# The real motivation for this theme is the possibility to test different heuristics at optimizing real-world-based non-trivial object function with local minima that is very easy to interpret. In other words, gain some experience with heuristics optimizing known function, so they can be trusted and successfully used at unknown later.
# 
# Moreover, the object function is parametric. If the player is one hundred percent accurate, the function is not continuous but contains several levels. In reality, there is some variance in player's throws and therefore, the shape of the function smooths to a mountain range like surface. An interesting question is how different heuristics will perform in both cases.

# ### Object Function
# The object function represents the dartboard and player's skills. It is not generally continuous, it contains approximately 30 evenly distributed local minima, and is limited, see fig below. We do not generally know its optimal value, but we can determine it for some values of its parameters.
# 
# It is defined on 2-dimensional, limited, continuous area, but for optimization purposes it is possible to lay 1mm grid, as player is not able to aim more precisely and the peaks will not be that sharp. It would made possible to evaluate the object function in approximately 40 thousand points.
# 
# ![object_function.jpg](attachment:object_function.jpg)

# ### Heuristic
# Good results require chosen heuristic to successfully identify the right peak that contains the optimal value. So reliability is the key metric here. However, evaluation of the object function requires computing an integral using Monte-Carlo method, which is time consuming. So the heuristic must make do with as little object function evaluation as possible, otherwise the required parametric analysis would be very computationally demanding or impossible at all.
# 
# Requirements:
# 1. Reliability: over 90 %.
# 2. Number of Evaluation: less than 1000.

# ### Workflow Outline:
# 1. Rewriting my object functions from MATLAB to course-specific python interface
# 2. Rewriting my dartboard plotter from MATLAB to python
# 3. Implementing general object for simpler experimenting with different kinds of heuristics at ones
# 4. Experimenting with heuristics from this course framework without exploitation detailed information about object functions
#    1. Testing heuristics on dartboard itself without tying the variance of the throw (left function on fig above)
#    2. Specifying variance of a very good player, calculating optimal value of such object function (right function on fig above)
#    3. Testing heuristics on just defined, smooth object function
#    4. Comparing heuristic performances on both variants of the object function
# 5. Trying to improve performance of the most promising heuristic for the smooth case by hyperparameter tuning 
# 6. Using the tuned heuristic for parametric analysis of optimal darts strategy
# 7. Discussing the results

# ### Imports

# In[1]:


# Import path to source directory
import sys
import os
pwd = get_ipython().run_line_magic('pwd', '')
sys.path.append(os.path.join(pwd, os.path.join('..','..', 'src')))

# Ensure modules are reloaded on any change
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[26]:


# Import extrenal libraries
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output

import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Import tool for convinient experimenting
from experimenting import ExperimentPerformer


# In[4]:


# Import object functions
from objfun_darts import Dartboard, DartsAvgScore
from objfun_darts import DartsPlotter


# In[5]:


# Import all available heuristics
from heur_aux import CauchyMutation, Correction, MirrorCorrection, ExtensionCorrection
from heur_sg import ShootAndGo
from heur_fsa import FastSimulatedAnnealing
from heur_de import DifferentialEvolution
from heur_go import GeneticOptimization
from heur_go import Crossover, UniformMultipoint, RandomCombination


# In[6]:


# Setting external libraries
sns.set()
pd.set_option('display.max_rows', 50)


# ### Definition of an overview experiment

# In[7]:


def get_overview_mutation(of):
    overview_mutations = []
    for correction in [Correction(of), MirrorCorrection(of)]:
            for r in [0.1, 0.5, 0.75]:
                overview_mutations.append(CauchyMutation(r, correction=Correction(of)))
    return overview_mutations


# In[8]:


def get_overview_heurs(of, maxeval, RD=True, FSA=True, DE=True, GO=True):
    overview_heurs = []

    if RD:
        for hmax in [0, 5, 10, 50, np.inf]:
            for RD in [False, True]:
                overview_heurs.append(ShootAndGo(of, maxeval=maxeval, hmax=hmax, random_descent=RD))
    if FSA:
        for T0 in [1e-10, 1e-2, 1, np.inf]:
            for mutation in get_overview_mutation(of):
                for n0 in [1, 2, 5]:
                    overview_heurs.append(FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=2, mutation=mutation))

    if DE:
        for N in [4, 10, 20]:
            for CR in [0.2, 0.5, 0.8]:
                for F in [0.5, 1, 2]:
                    overview_heurs.append(DifferentialEvolution(of, maxeval=maxeval, N=N, CR=CR, F=F))

    if GO:
        for mutation in get_overview_mutation(of):
            for crossover in [Crossover(), UniformMultipoint(1), RandomCombination()]:
                for N in [1, 2, 5, 10, 30, 100]:
                    for Tsel1 in [0.5, 1]:
                        overview_heurs.append(GeneticOptimization(of, maxeval, N=N, M=N*3, Tsel1=Tsel1, Tsel2=0.1, 
                                                                  mutation=mutation,
                                                                  crossover=crossover))   
    return overview_heurs


# In[18]:


def run_experiment(of, performer, heurs, num_runs):
    results = performer.experiment(heurs, num_runs)
    print('Best performance:')
    display(performer.get_stats().head(25))
    print('Worst performance:')
    display(performer.get_stats().tail(25))
    return results


# ### Overview Experiments
# #### Warning! Running "run_overview_experiment" for object function "avg_score" will take atleast an hour on PC.

# In[19]:


NUM_RUNS = 20
MAX_EVAL = 1000


# #### 1. Multilevel object function - Dartboard 

# In[10]:


dartboard = Dartboard()
dartboard_performer = ExperimentPerformer('dartboard.log')


# In[ ]:


overview_heurs = get_overview_heurs(dartboard, MAX_EVAL)


# In[11]:


dartboard_results = run_experiment(dartboard, dartboard_performer, overview_heurs, NUM_RUNS)


# #### 2. Smooth object function - Average Score

# In[12]:


avg_score = DartsAvgScore(7)  # 7mm... Superhuman trow variance
avg_score.fstar = -52.5  # Only peek at triple 20 reaches greater values
score_performer = ExperimentPerformer('avg_score.log')


# In[ ]:


overview_score_heurs = get_overview_heurs(avg_score, MAX_EVAL)


# In[13]:


score_results = run_experiment(avg_score, score_performer, overview_score_heurs, NUM_RUNS)


# #### Performance Evaluation
# Generic methods and simulated annealing turned out badly for all tested parameters. Intuitively, it does not make much sense to mutate or crossover coordinates of 2D surface points and hope it will identify the right peak. But the main reason could be the value of Couchy Mutation parameter "r", that was set to value lower than 1 in all cases. It results into many small steps that probably wasted precious evaluations.  
# 
# Due to the relatively large area with the optimal value, Random Shooting was very successful. As expected, it performed better than the other local-gradient-based Shoot and Go methods during optimizing the plane Dartboard object function. However, it is a bit disappointing that this was also the case in the second version of the object function. Better results from Random and Steepest descent were expected there. Perhaps if neighborhood was defined differently (larger and thinner), the results could be better.
# 
# The only heuristic that could compete with Random Shooting in reliability is Differential Evolution. This algorithm was absolutely the worst with a small population, but larger value of the parameter N improved significantly its performance. The effects of the other parameters are not clear yet. The reason for the success of the algorithm will probably be an even initial distribution of its agents (Random Shooting) and relatively large jumps that the points can make in one iteration.
# 
# Altogether, the results of the initial tests are not entirely encouraging. One of the two most successful methods is just randomly selecting points. Most sophisticated heuristics have burned out. The only hope seems to be to try to optimize the performance of Differential Evolution, but it seems that the main reason for this heuristic success is its similarity to Random Shooting.

# ### Corrections

# #### Last chance for GO and FSA

# In[20]:


def get_correction_mutation(of):
    correction_mutations = []
    for correction in [Correction(of), MirrorCorrection(of)]:
            for r in [1, 1.5, 2, 5]:
                correction_mutations.append(CauchyMutation(r, correction=Correction(of)))
    return correction_mutations


# In[21]:


def get_correction_heurs(of, maxeval, FSA=True, GO=True):
    overview_heurs = []

    if FSA:
        for T0 in [1e-10, 1e-2, 1, np.inf]:
            for mutation in get_correction_mutation(of):
                for n0 in [1, 2, 5]:
                    overview_heurs.append(FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=2, mutation=mutation))
    if GO:
        for mutation in get_correction_mutation(of):
            for crossover in [UniformMultipoint(1)]:
                for N in [100, 150, 200]:
                    for Tsel1 in [0.5]:
                        overview_heurs.append(GeneticOptimization(of, maxeval, N=N, M=N*3, Tsel1=Tsel1, Tsel2=0.1, 
                                                                  mutation=mutation,
                                                                  crossover=crossover))   
    return overview_heurs


# In[22]:


correction_heurs = get_correction_heurs(avg_score, MAX_EVAL)


# In[24]:


correction_results = run_experiment(avg_score, score_performer, correction_heurs, NUM_RUNS)


# #### Evaluation
# No major improvement was observed after the expansion of the parameter space of heuristics GO and FSA. Tuning DE is still the only hope for the successful fulfillment of the formal goal of the work.

# #### Tuning Differential Evolution

# #### 1. Parameter Space Expansion

# In[40]:


def get_de_results_for_tuning(of, performer, num_runs, maxeval):
    results = pd.DataFrame()
    
    for N in [10, 15, 20, 25, 50]:
        for CR in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for F in [0.1, 0.5, 1, 1.5, 2]:
                heur = DifferentialEvolution(of, maxeval=maxeval, N=N, CR=CR, F=F)
                result = performer.experiment(heur, num_runs)
                clear_output(wait=True)
                result['N'] = N
                result['CR'] = CR
                result['F'] = F
                results = pd.concat([results, result], axis=0)
    return results


# In[41]:


table = get_de_results_for_tuning(avg_score, score_performer, NUM_RUNS, MAX_EVAL)
display(table)
table.to_csv('de_tuning.log', header=True)


# In[42]:


score_performer.get_stats().head(25)


# #### Evaluation
# The expansion of the parameter space indicated that a large population could indeed have a positive effect on performance. For other parameters, their influence is still not clear. However, CR values around 0.5 and F less than 1 seems to have the best results. Further research is needed.

# #### 2. Parameter Analysis 

# In[104]:


def analyze_parameter(table, param_name, value_name='best_y'):
    display(table.groupby(param_name).agg({value_name: ['mean', 'median','var', 'max']}))
    if not isinstance(param_name, list):
        plt.figure()
        sns.boxplot(x=param_name, y=value_name, data=table)


# In[97]:


analyze_parameter(table, 'N')


# These results for param N confirm the assumed fact that population growth has a positive effect on performance. However, the mean values for each population are quite small. It means other parameters do have significant effect. Further expansion of the parametric space, in order to find the optimal population is also necessary.

# In[98]:


analyze_parameter(table, 'CR')


# This is quite surprising. The best average results are for CR value 0, which is in conflict with overview experiments, during which the best places usually were taken by combinations with CR around 0.5. It is possible that these go better along larger populations. However, the values of the statistics for 0 are comparable to those for 0.4, which has even the highest median. However, following table can show the situation better.

# In[109]:


analyze_parameter(table, ['CR', 'N'])


# Small values of the CR parameter complement well with a medium-sized population, especially points (0, 20) and (0.2, 25) are interesting. Values around 0.5 again work well with a larger population. These combinations tend to be more reliable e.g. (0.4, 50), (0.6, 50).

# In[99]:


analyze_parameter(table, 'F')


# In[106]:


analyze_parameter(table, ['F', 'N'])


# For all possible population sizes, the optimal value for F seems to be 0.5. The neighborhood of this point should be further explored.

# #### Parameters adjustment

# In[112]:


def get_de_results_for_tuning_adjustment(of, performer, num_runs, maxeval):
    results = pd.DataFrame()
    
    for N in [30, 40, 60, 80, 100]:
        for CR in [0.4, 0.5, 0.6]:
            for F in [0.4, 0.5, 0.6]:
                heur = DifferentialEvolution(of, maxeval=maxeval, N=N, CR=CR, F=F)
                result = performer.experiment(heur, num_runs)
                clear_output(wait=True)
                result['N'] = N
                result['CR'] = CR
                result['F'] = F
                results = pd.concat([results, result], axis=0)
    return results


# In[113]:


table_adj = get_de_results_for_tuning_adjustment(avg_score, score_performer, NUM_RUNS, MAX_EVAL)
table_adj.to_csv('de_tuning.log', mode='a', header=False)
table_adj['best_y'] = table_adj['best_y'].apply(lambda x: x[0])
all_table = pd.concat([table, table_adj], axis=0)
score_performer.get_stats().head(15)


# Promising results have finally arrived. Especially the combination 30-0.5-0.4 looks very promising, but more runs are. However, it is necessary to perform additional tests on more runs.

# #### Selecting the most promising variants
# 
# The following parameter analysis is intended to identify parameter combinations that will be further tested on a larger number of runs.

# In[115]:


analyze_parameter(table_adj, 'N')


# In[116]:


analyze_parameter(table_adj, 'F')


# In[117]:


analyze_parameter(table_adj, 'CR')


# #### Testing the most promising variants

# In[118]:


def test_de_variant(of, performer, maxeval, N, CR, F, num_run):
    heur = DifferentialEvolution(of, maxeval=maxeval, N=N, CR=CR, F=F)
    result = performer.experiment(heur, num_runs)
    clear_output(wait=True)
    result['N'] = N
    result['CR'] = CR
    result['F'] = F
    return result

def test_most_promising_variants(of, performer, maxeval):
    num_runs = 100
    results = pd.DataFrame()
       
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 20, 0, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 20, 0.5, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 25, 0.2, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 30, 0.5, 0.4, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 30, 0.4, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 40, 0.4, 0.6, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 50, 0.4, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 50, 0.6, 0.5, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 60, 0.4, 0.6, num_run)], axis=0)
    results = pd.concat([results, test_de_variant(of, performer, maxeval, 100, 0.6, 0.6, num_run)], axis=0)

    return results


# In[120]:


table_best = get_de_results_for_tuning_adjustment(avg_score, score_performer, NUM_RUNS, MAX_EVAL)
table_best.to_csv('de_tuning.log', mode='a', header=False)
table_best['best_y'] = table_best['best_y'].apply(lambda x: x[0])
all_table = pd.concat([all_table, table_best], axis=0)
score_performer.get_stats().head(15)


# In[121]:


analyze_parameter(table_best, 'N')


# #### The best DE varinat
# As the best variant of differential evolution, parameters N = 30, CR = 0.5, F = 0.4  were chosen. This combination was significantly more reliable than other alternatives and achieved decent results in all other monitored metrics. However, even this option does not fully meet the original requirements.

# In[123]:


best_N = 30
best_CR = 0.5
best_F = 0.4


# ### Optimal Strategies

# In[164]:


def find_optimal_strategies():
    optimal_strategies = {'variability': [], 'best_y': [], 'best_x': []}
    for variability in tqdm(range(0, 100, 2)):
        of = DartsAvgScore(variability)
        heur = DifferentialEvolution(of, maxeval=1000, N=best_N, CR=best_CR, F=best_F)
        result = heur.search()
        optimal_strategies['variability'].append(variability)
        optimal_strategies['best_y'].append(result['best_y'])
        optimal_strategies['best_x'].append(result['best_x'])
    return optimal_strategies


# In[168]:


optimal_strategies = find_optimal_strategies()


# In[169]:


points_for_plotter = list(map(list, zip(*optimal_strategies['best_x'])))


# In[170]:


plotter = DartsPlotter()
plotter.plot_points(dartboard, points_for_plotter, optimal_strategies['best_y'])


# #### Original Result
# Optained by flock-based algorithm SPSO_2006 during the previous seminar work on MMC.
# ![original_solution.png](attachment:original_solution.png)

# ### Summary
# All heuristics from this-course framework were tested on optimizing two versions of object function, standard dartboard and its smooth average-score version.
# 
# Unfortunately, no large differences in the performance of individual heuristics were observed between both object functions. Only the smooth version was more difficult to optimize, as the area of the optimal value was smaller.
# 
# The only heuristic that managed to overcome Random Shooting in both basic cases was Differential Evolution. This heuristic was then further investigated. Its reliability has been improved by 15 percent to almost 90 percent by hyper-parameter tuning. 
# 
# The best version of DE was then applied to find the optimal strategy of playing darts depending on the player's ability to hit the place he is aiming for. Heuristic was able to identify the two most important peaks, but it did not prove the correct course of the strategy, despite all efforts. Far better results were achieved by the heuristic SPSO_2006 in the original work.
