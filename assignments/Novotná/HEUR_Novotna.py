#!/usr/bin/env python
# coding: utf-8

# # Continuous functions minimization by dynamic random search technique (DRASET)
# 
# This works presents an algorithm DRASET for continuous functions. DRASET was first introduced in this article: https://www.sciencedirect.com/science/article/pii/S0307904X06002071 All steps of DRASET can be found there. The main ideas of DRASET are:
# * there are two stages of the algorithm: the general search phase and the local search phase
# * in the general search phase the algorithm tries to search as much space as possible and remembers the best found solution
# * in the local search phase algorithm tries to improve the best solution found in the general search phase
# * in the both phases the algorithm works with the best found solution and the current solution
# * a random vector generated from a range $\langle-\alpha, \alpha \rangle$ is added or substracted from the current solution and the function is evaluated at this number
# * then there are some decisions, for more details see the article
# * important is that if a better solution than the current best solution is found, the current solution doesn't change - it prevents from stucking in a local optimum
# * in the general search phase alpha differs every $N_{general}$ steps and in the local search phase alpha differs every $N_{local}$ steps
# * $E$ (epochs) represents the number of generating the random vector
# * the general search phase lasts $E - 2*N_{general}$ steps
# * the local search phase lasts $2*N_{general}$ steps
# * the number of epochs doesn't equal the number of function evaluations, in the worst case the function can be evaluated for $2*E$ times

# In[1]:


# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().run_line_magic('pwd', '')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from objfun_plato import Plato
from objfun_banana import Banana
from heur_draset import Draset
from heur_aux import Correction
from heur_sg import ShootAndGo


# In[3]:


np.random.seed(6)


# We will study DRASET's behavior on two functions - Rastrigin's function 6 and Rosenbrock's valley (De Jong's function 2). Both functions are defined here: http://www.geatbx.com/docu/fcnindex-01.html There are also graphs of these functions.
# 
# Random shooting is chosen as a baseline method. 

# We start with Rastrigin's function 6. This function has many local minima.

# In[4]:


function = Plato(n = 2, eps = 0.1)
corr = Correction(function)  # correction is not discussed in the original article, I choose the simpliest one


# In[5]:


RUNS = 500
MAXEV = 50000  # the heuristic is implemented with maxeval parameter, I don't want to restrict heuristic by this parameter, so I set it at this high level


# We define two experiments: DRASET and random shooting.

# In[6]:


def experiment_draset(of, maxeval, num_runs, E, alfa0, N_general, N_local, correction): 
    results = []
    for i in tqdm(range(num_runs), 'Testing E={}, alfa0={}, N_general={}, N_local={}'.format(E, alfa0, N_general, N_local)):
        result = Draset(of, maxeval, E, alfa0, N_general, N_local, correction).search()
        result['run'] = i
        result['heur'] = 'Draset_{}_{}_{}_{}'.format(E, alfa0, N_general, N_local)
        result['E'] = E
        result['alfa0'] = alfa0
        result['N_general'] = N_general
        result['N_local'] = N_local
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'E', 'alfa0', 'N_general', 'N_local', 'best_x', 'best_y', 'neval', 'epoch', 'alfa', 'f_x', 'log_data'])


# In[7]:


def experiment_random(of, maxeval, num_runs):
    results = []
    for i in tqdm(range(num_runs), 'Testing maxeval={}'.format(maxeval)):
        result = ShootAndGo(of, maxeval, hmax = 0).search()
        result['run'] = i
        result['heur'] = 'Random_{}'.format(maxeval)
        result['maxeval'] = maxeval
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'maxeval', 'best_x', 'best_y', 'neval'])


# We define statistics for a later analysis.

# In[8]:


def rel(x):
    return len([n for n in x if n < np.inf])/len(x)

def mne(x):
    return np.mean([n for n in x if n < np.inf])

def feo(x):
    return mne(x)/rel(x)


# In[9]:


def mean(x):
    return np.mean(x)

def med(x):
    return np.median(x)


# Now we run the first experiment. In the article initial $\alpha = 1$ and $N_{local} = \frac{1}{5} N_{general}$ setting was recomended.

# In[10]:


table_draset = pd.DataFrame()

for E in [250, 500, 1000, 2000, 5000]:
    for alfa0 in [1]:
        for N_general in [50, 100, 500, 1000, 2000]:
            if 2*N_general < E:  # we will run only settings which make sense
                N_local = N_general/5
                res = experiment_draset(of = function, maxeval = MAXEV, num_runs = RUNS, E = E, alfa0 = alfa0, N_general = N_general, N_local = N_local, correction = corr)
                table_draset = pd.concat([table_draset, res], axis = 0)
    


# In[11]:


stats_draset = table_draset.pivot_table(
    index=['heur', 'E', 'alfa0', 'N_general', 'N_local'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_draset = stats_draset.reset_index()


# In[12]:


stats_draset.sort_values(by=['feo'])


# The best Feoktistov criterion is obtained for $E = 2000$. 

# In[13]:


stats_draset.sort_values(by=['rel'], ascending = False)


# We can see a clear pattern that for more epochs we get higher reliability. But generally we can't say anything about a relation between the number of epochs and $N_{general}$ and better or worse reliability or Feoktistov criterion.

# Now we focus on the best found value of the function after given number of epochs. The optimum equals 0.

# In[14]:


stats_draset_best = table_draset.pivot_table(
    index=['heur', 'E', 'alfa0', 'N_general', 'N_local'],
    values=['best_y'],
    aggfunc=(mean, med)
)['best_y']
stats_draset_best = stats_draset_best.reset_index()


# In[15]:


stats_draset_best.sort_values(by=['med'])


# The table is sorted by median because it is a more robust statistic. We can see that with more epochs the algorithm finds a solution closer to the optimum.

# Now we run the random shooting algorithm. We set a maximum number of evaluations of the function as $2*E$ because, as it was said before, in the worst case during DRASET the function can be evaluated $2*E$ times.

# In[16]:


table_random = pd.DataFrame()

for maxeval in [500, 1000, 2000, 4000, 10000]:
    res = experiment_random(of = function, maxeval = maxeval, num_runs = RUNS)
    table_random = pd.concat([table_random, res], axis = 0)


# In[17]:


stats_random = table_random.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_random = stats_random.reset_index()


# In[18]:


stats_random.sort_values(by=['rel'], ascending = False)


# It is obvious that we get much worse results than by DRASET. Both Feoktistov criterion and reliability are worse.

# In[19]:


stats_random_best = table_random.pivot_table(
    index=['heur'],
    values=['best_y'],
    aggfunc=(mean, med)
)['best_y']
stats_random_best = stats_random_best.reset_index()


# In[20]:


stats_random_best.sort_values(by=['med'])


# The best found solution is also worse than the solution found by DRASET.

# Now we analyze the second function: Rosenbrock's valley (De Jong's function 2). The global optimum is inside a long, narrow, parabolic shaped flat valley and equals 0.

# In[21]:


function = Banana(n = 2, eps = 0.01)  # In this case we want to find more precise solution.
corr = Correction(function)  # correction is not discussed in the original article, I choose the simpliest one


# First we run DRASET experiment.

# In[22]:


table_draset2 = pd.DataFrame()

for E in [250, 500, 1000, 2000, 5000]:
    for alfa0 in [1]:
        for N_general in [50, 100, 500, 1000, 2000]:
            if 2*N_general < E:  # we will run only settings which make sense
                N_local = N_general/5
                res = experiment_draset(of = function, maxeval = MAXEV, num_runs = RUNS, E = E, alfa0 = alfa0, N_general = N_general, N_local = N_local, correction = corr)
                table_draset2 = pd.concat([table_draset2, res], axis = 0)
    


# In[23]:


stats_draset2 = table_draset2.pivot_table(
    index=['heur', 'E', 'alfa0', 'N_general', 'N_local'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_draset2 = stats_draset2.reset_index()


# In[24]:


stats_draset2.sort_values(by=['feo'])


# Acording to Feoktistov criterion 250 epochs and $N_{general}$ = 100 is the best setting for DRASET. But reliability of this setting isn't high. For $E = 2000$ or $5000$ Feoktistov criterion is worse than for less epochs because of more function evaluations.

# In[25]:


stats_draset2.sort_values(by=['rel'], ascending = False)


# We can see a clear pattern that for more epochs we get a higher reliability. 

# In[26]:


stats_draset_best2 = table_draset2.pivot_table(
    index=['heur', 'E', 'alfa0', 'N_general', 'N_local'],
    values=['best_y'],
    aggfunc=(mean, med)
)['best_y']
stats_draset_best2 = stats_draset_best2.reset_index()


# In[27]:


stats_draset_best2.sort_values(by=['med'])


# The table is sorted by median because it is a more robust statistic. We can see that with more epochs (only with one exception) the algorithm finds a solution closer to the optimum.

# Now we run the random shooting algorithm. We set a maximum number of evaluations of the function as $2*E$ because, as it was said before, in the worst case during DRASET the function can be evaluated $2*E$ times.

# In[28]:


table_random2 = pd.DataFrame()

for maxeval in [500, 1000, 2000, 4000, 10000]:
    res = experiment_random(of = function, maxeval = maxeval, num_runs = RUNS)
    table_random2 = pd.concat([table_random2, res], axis = 0)


# In[29]:


stats_random2 = table_random2.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_random2 = stats_random2.reset_index()


# In[30]:


stats_random2.sort_values(by=['rel'], ascending = False)


# In[31]:


stats_random_best2 = table_random2.pivot_table(
    index=['heur'],
    values=['best_y'],
    aggfunc=(mean, med)
)['best_y']
stats_random_best2 = stats_random_best2.reset_index()


# In[32]:


stats_random_best2.sort_values(by=['med'])


# We again get worse results than by DRASET. 

# ## Conclusion
# If we compare random shooting and DRASET, it is obvious that DRASET is a better algorithm than random shooting. DRASET has lower Feoktistov criterion, a higher reliability and the best found solution is closer to the optimum. If we care about the reliability or finding the closest solution, I recommend to run thousends of epochs. If we care only about Feoktistov criterion, it is difficult to recommend the optimal setting. Probably it would be less epochs than for the best reliability. I have no recommendations about $N_{general}$. There is no visible influence of it's value. Number of epochs is a more important parameter.
