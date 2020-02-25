
# coding: utf-8

# # Lecture outline
# 
# 1. Resources for learning Python
# 1. Experimental framework implementation
# 1. Performance evaluation
# 1. Assignments

# # 1. Resources for learning Python, numpy, pandas and Jupyter notebooks
# 
# **Read this guide/tutorial first**: [Running Jupyter Notebooks With The Anaconda Python Distribution](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook) and please note that:
# * You can skip parts on pip & docker. 
# * Install Anaconda (with Python 3) only.
# * Do not hesitate to follow links included in the guide.
# 
# **Then**, consider proceeding to:
# 
# 1. [Intro to Python for Data Science](https://www.datacamp.com/courses/intro-to-python-for-data-science) - I do not have personal experience with this course, but since it was referenced in the DataCamp guide, it should be OK (and free)
# 1. [Python Data Science Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/python-data-science-cookbook) - book I can highly recommend
# 1. [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/) - Good Udemy course with examples in Jupyter notebooks (often in sale for ca. $10)
# 1. Last, but not least, use *Google*. Feel free to type in name of module you are interested in and you will find lots of helpful resources. Official documentation sites (e.g. for [NumPy](https://docs.scipy.org/doc/numpy/reference/)) are great places to start as well.

# **Good luck!**
# 
# Little bit of motivation:
# 
# <img src="img/matlab_vs_python_2020.png">
# 
# Source: [Google Trends](https://trends.google.com/trends/explore?date=all&q=%2Fm%2F053_x,%2Fm%2F05z1_)
# 
# Disclaimer: _I am very happy user of MATLAB, as well_ :-)

# # 2. Experimental framework implementation
# 
# ## Best practice to implement and test $n$ heuristics and $m$ objective functions?
# 
# * There are some common characteristics for the two most important _things_ in our framework
#   * Heuristics - store the best found solution, manage stop criterion, etc.
#   * Objective functions - store $f^*$, lower/upper bounds, etc.
# * Every specific heuristic or obj. function implements its own search space exploration or evaluation, neighbourhood generation, etc.
# * Thus, the object-oriented design should help us to separate these concerns as much as possible and also to keep us sane.
# 
# <img src="img/oop_design.png">

# ## Example: two objective functions and generalized Shoot&Go
# 
# 
# ### Objective functions
# 
# #### 1. AirShip
# 
# * Same as on the previous class, i.e. we will **minimize** obj. function values
# * implemented as ``class AirShip(ObjFun)`` in ``src/objfun_airship.py``

# In[1]:

# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

from objfun_airship import AirShip
airship = AirShip()


# In[3]:

airship.get_bounds()


# In[4]:

airship.get_fstar()


# In[5]:

airship.generate_point()


# In[6]:

airship.evaluate(50)


# #### 2. $\sum \mathbf{x}$
# 
# * Just as demonstration of vectorized lower/upper bounds
# * implemented as ``class Sum(ObjFun)`` in ``src/objfun_sum.py``

# In[7]:

from objfun_sum import Sum
of_sum = Sum([0, 0, 0, 0], [10, 10, 10, 10])


# In[8]:

x = of_sum.generate_point()
print(x)
print(of_sum.evaluate(x))


# In[9]:

print(of_sum.get_neighborhood(x, 1))


# In[10]:

print(of_sum.get_neighborhood(x, 2))


# ^^ This behaviour is intended. See code for details.

# In[11]:

of_sum.get_neighborhood([0, 0, 0, 0], 1)


# ### Generalized Shoot&Go: $\mathrm{SG}_{hmax}$
# 
# * Shoot & Go heuristic (also known as *Iterated Local Search*, *Random-restart hill climbing*, etc)
#     * $hmax \in \{ 0, 1, \ldots, \infty \}$ parameter - maximum number of local searches / hill climbs
#     * note that $\mathrm{SG}_{0}$ is pure Random Shooting (Random Search)
#     
# * implemented as ``class ShootAndGo(Heuristic)`` in ``src/heur_sg.py``    
# 
# 

# In[12]:

from heur_sg import ShootAndGo


# In[13]:

# Random Shooting for the AirShip initialization...
demo_rs = ShootAndGo(airship, maxeval=100, hmax=0)
# ...and execution:
demo_rs.search()


# # 2. Performance evaluation
# 
# ## What is the recommended approach to store and analyze results of your experiments?
# 
# 1. Append all relevant statistics from a single run into table (e.g. CSV file in memory or on disk), including all task and heuristic parameters 
# 2. Load the table into analytical tool of your choice (**data frame**, Excel or Google Docs spreadsheets, etc.)
# 3. Pivot by relevant parameters, visualize in tables or charts

# ## Demonstration
# 
# Neccessary setup first:

# In[14]:

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


# ### General experiment setup
# 
# Runs selected objective function (`of`) using selected heuristic multiple times, stores and returns data (results) in a data frame.

# In[15]:

def experiment(of, num_runs, hmax):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing hmax = {}'.format(hmax)):
        result = ShootAndGo(of, maxeval=100, hmax=hmax).search() # dict with results of one run
        result['run'] = i
        result['heur'] = 'SG_{}'.format(hmax) # name of the heuristic
        result['hmax'] = hmax
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'hmax', 'best_x', 'best_y', 'neval'])


# ### Air Ship experiments

# In[16]:

table = pd.DataFrame()
for hmax in [0, 1, 2, 5, 10, 20, 50, np.inf]:
    res = experiment(airship, 10000, hmax)
    table = pd.concat([table, res], axis=0)


# **Note**: This is what you should see while experiments are in progress:
# 
# <img src="img/tqdm_progress.png">

# In[17]:

table.info()


# In[18]:

table.head()


# #### What is the quality of solutions based on `hmax`?
# 
# In **tabular** form:

# In[19]:

table.groupby(['hmax'])['best_y'].median()


# In[20]:

table.groupby(['hmax'])['best_y'].mean()


# Feel free to compute other statistics instead of median and mean.
# 
# Directly as **Box-Whiskers plot**:

# In[21]:

# import visualization libraries
import matplotlib
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[22]:

ax = sns.boxplot(x="hmax", y="best_y", data=table)


# #### Number of evaluations (when successful), based on `hmax`?
# 
# Let's add another column, `success`:

# In[23]:

table['success'] = table['neval'] < np.inf


# In[24]:

table[table['success'] == True].head()


# Table:

# In[25]:

table[table['success'] == True].groupby(['hmax'])['neval'].mean()


# In[26]:

table[table['success'] == True].groupby(['hmax'])['neval'].median()


# Chart:

# In[27]:

ax = sns.boxplot(x="hmax", y="neval", data=table[table['success'] == True])


# #### Reliability

# In[28]:

rel_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: len([n for n in x if n < np.inf])/len(x)
)


# In[29]:

rel_by_hmax


# In[30]:

ax = rel_by_hmax.plot(kind='bar')


# #### Speed, normalized by reliability?
# 
# * Reliability: $REL = m/q$ where $m$ is number of successful runs and $q$ is total number of runs, $REL \in [0, 1]$
# * Mean Number of objective function Evaluations: $MNE = \frac{1}{m} \sum_{i=1}^m neval_i$
# * Feoktistov criterion: $FEO = MNE/REL$

# In[31]:

feo_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: np.mean([n for n in x if n < np.inf])/(len([n for n in x if n < np.inf])/len(x))
    #                 ^^^   mean number of evaluations ^^^ / ^^^             reliability         ^^^^
)


# In[32]:

feo_by_hmax


# In[33]:

ax = feo_by_hmax.plot(kind='bar')


# ### `sum(x)` experiments

# In[34]:

table = pd.DataFrame()
for hmax in [0, 1, 2, 5, 10, 20, 50, np.inf]:
    res = experiment(of_sum, 10000, hmax)
    table = pd.concat([table, res], axis=0)


# #### Quality of solutions based on hmax?

# In[35]:

ax = sns.boxplot(x="hmax", y="best_y", data=table)


# #### Number of evaluations (when successful), based on hmax?

# In[36]:

table['success'] = table['neval'] < np.inf


# In[37]:

ax = sns.boxplot(x="hmax", y="neval", data=table[table['success'] == True])


# #### Reliability?

# In[38]:

rel_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: len([n for n in x if n < np.inf])/len(x)
)


# In[39]:

rel_by_hmax


# In[40]:

ax = rel_by_hmax.plot(kind='bar')


# #### Feoktistov criterion?

# In[41]:

feo_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: np.mean([n for n in x if n < np.inf])/(len([n for n in x if n < np.inf])/len(x))
)


# In[42]:

ax = feo_by_hmax.plot(kind='bar')


# # Assignments (or rather inspiration for your own work)
# 
# 1. Implement examples in this notebook on your own
# 1. Experiment with **neighbourhood diameter** `d` in `AirShip.get_neighborhood(x, d)`
# 1. Play with new heuristics in the existing framework and analyze their performance:
#    1. **Random Descent**, similar to Shoot & Go, but does not follow steepest descent, chooses direction of the descent randomly instead
#    1. **Taboo Search**
