#!/usr/bin/env python
# coding: utf-8

# # Genetic Optimization (Continuation)
# 
# Before we start, please revise example GO implementation in `src/heur_go.py`. Ideally, compare it to your solution!
# 
# This notebok will give you another ideas for crossover operators and mutation correction strategies.

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


# Import external librarires
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# **Let's use the well-known ``TSPGrid(3, 3)`` for demonstration purposes**

# In[3]:


from objfun_tsp import TSPGrid
tsp = TSPGrid(3, 3)


# # Three different crossover operators:

# **First, let's assume these are our parents:**

# In[4]:


x = np.zeros(10, dtype=int)
x


# In[5]:


y = 9*np.ones(10, dtype=int)
y


# ## 1. Random mix (baseline class)

# In[6]:


from heur_go import Crossover
co_rnd = Crossover()
co_rnd.crossover(x, y)


# ## 2. Uniform n-point crossover

# In[7]:


from heur_go import UniformMultipoint
co_uni = UniformMultipoint(4)
co_uni.crossover(x, y)


# ## 3. Random combination

# In[8]:


from heur_go import RandomCombination
co_comb = RandomCombination()
co_comb.crossover(x, y)


# # Demonstration

# In[9]:


from heur_go import GeneticOptimization
from heur_aux import CauchyMutation, Correction


# In[10]:


NUM_RUNS = 1000
maxeval = 1000


# In[11]:


# prepare battery of crossovers to be tested (with some metadata)
crossovers = [
    {'crossover': Crossover(), 'name': 'mix'},
    {'crossover': UniformMultipoint(1), 'name': 'uni'},  #  test for other n as well!
    {'crossover': RandomCombination(), 'name': 'rnd'},
]


# In[12]:


results = pd.DataFrame()
for crossover in crossovers:
    heur_name = 'GO_{}'.format(crossover['name'])
    runs = []
    for i in tqdm(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = GeneticOptimization(tsp, maxeval, N=5, M=15, Tsel1=1, Tsel2=0.1, 
                                  mutation=CauchyMutation(r=.75, correction=Correction(tsp)),
                                  crossover=crossover['crossover']).search()
        run['run'] = i
        run['heur'] = heur_name
        run['crossover'] = crossover['name']
        runs.append(run)
    
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'crossover', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)


# In[13]:


def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[14]:


results_pivot = results.pivot_table(
    index=['heur', 'crossover'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='crossover')


# #### Conclusion
# 
# It seems that the alternative crossover methods are not improving performance in this case.

# ## Assignments
# 
# * Thoroughly test different kinds of GO setup, also using the new corrections strategies (`MirrorCorrection` and `ExtensionCorrection`, see `src/heur_aux.py`)
# * Could you think of any other crossover operator? See e.g. https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm) for inspiration
# * Implement and optimize the following objective function...

# ### Clerc's Zebra-3 objective function
# 
# Clerc's Zebra-3 problem is a non-trivial binary optimization problem and part of discrete optimization benchmark problems (Hierarchical swarm model: a new approach to optimization, Chen et al, 2010).
# 
# Zebra-3 function is defined for $d = 3 \, d^*$, $d^* \in \mathbb{N}$ as
# $$ \mathrm{z}(\boldsymbol{\mathsf{x}}) = \sum_{k=1}^{d^*} \mathrm{z}_{1+\mathrm{mod}(k-1,2)} (\boldsymbol{\mathsf{\xi}}_k) $$
# where
# $\boldsymbol{\mathsf{\xi}}_k = (x_{3\,k-2}, \ldots, x_{3\,k})$ and
# 
# $$
# \mathrm{z_1}(\boldsymbol{\mathsf{\xi}}) = \left\{
# \begin{array}{c l}     
#     0.9 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=0 \\
#     0.6 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=1 \\
#     0.3 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=2 \\
#     1.0 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=3  
# \end{array}\right.
# $$
# 
# $$
# \mathrm{z_2}(\boldsymbol{\mathsf{\xi}}) = \left\{
# \begin{array}{c l}     
#     0.9 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=3 \\
#     0.6 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=2 \\
#     0.3 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=1 \\
#     1.0 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=0 
# \end{array}\right.
# $$
# 
# Zebra-3 function is a subject of maximization with maximum value of $d/3$. Therefore  we will minimize 
# 
# $$\mathrm{f}(\boldsymbol{\mathsf{x}})=\frac{d}{3} - \mathrm{z}(\boldsymbol{\mathsf{x}})$$
# 
# with $f^* = 0$.

# Illustration in 3D:
# 
# <img src="img/zebra3.png">

# Implementation: see `src/objfun_zebra3.py`.
