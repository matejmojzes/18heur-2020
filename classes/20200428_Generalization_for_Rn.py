#!/usr/bin/env python
# coding: utf-8

# # Generalization for $\mathbb{R}^n$
# 
# Our framework should automatically recognize objective funcion domain and use proper routines for each domain ($\mathbb{Z}^n$ or $\mathbb{R}^n$).

# ### Set up IPython notebook environment first...

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


# Import external libraries
import numpy as np


# ## Testing `numpy.dtype`

# In[3]:


zn = np.ones(10, dtype=int)
zn


# In[4]:


zn.dtype


# In[5]:


rn = np.ones(10)
rn


# In[6]:


rn.dtype


# In[7]:


from heur_aux import is_integer


# In[8]:


is_integer(rn)


# In[9]:


is_integer(zn)


# ## De Jong 1 objective function
# 
# Source: http://www.geatbx.com/docu/fcnindex-01.html#P89_3085

# In[10]:


from objfun_dejong1 import DeJong1  # substatial difference to existing functions is the epsilon parameter!


# In[11]:


dj = DeJong1(n=3, eps=0.1)


# In[12]:


dj.a


# In[13]:


dj.b


# In[14]:


x = dj.generate_point()
x


# In[15]:


dj.evaluate(x)


# In[16]:


# optimum
dj.evaluate(np.zeros(5))


# ## Generalized mutation demo on De Jong 1
# 
# Let's test mutation corrections first:

# In[17]:


from heur_aux import Correction, MirrorCorrection, ExtensionCorrection


# In[18]:


# sticky correction in R^n (mind x[1])
Correction(dj).correct(np.array([6.12, -4.38,  2.96]))


# In[19]:


# mirror correction in R^n (mind x[1])
MirrorCorrection(dj).correct(np.array([6.12, -4.38,  2.96]))


# In[20]:


# extension correction in R^n (mind x[1])
ExtensionCorrection(dj).correct(np.array([6.12, -4.38,  2.96]))


# I.e. corrections work also in the continuous case, as expected...

# In[21]:


from heur_aux import CauchyMutation


# In[22]:


cauchy = CauchyMutation(r=.1, correction=MirrorCorrection(dj))
cauchy.mutate(np.array([6.12, -4.38,  2.96]))


# ## De Jong 1 optimization via FSA
# 
# Thanks to current state of the framework, no modification to FSA is needed.

# In[23]:


from heur_fsa import FastSimulatedAnnealing


# In[24]:


heur = FastSimulatedAnnealing(dj, maxeval=10000, T0=10, n0=10, alpha=2, 
                              mutation=cauchy)
res = heur.search()
print(res['best_x'])
print(res['best_y'])
print(res['neval'])


# ## De Jong 1 optimization via GO
# 
# Let's review modified crossover operators in $\mathbb{R}^n$ first:

# In[25]:


from heur_go import Crossover, UniformMultipoint, RandomCombination


# In[26]:


x = dj.generate_point()
y = dj.generate_point()
print(x)
print(y)


# In[27]:


Crossover().crossover(x, y)


# In[28]:


UniformMultipoint(1).crossover(x, y)


# In[29]:


RandomCombination().crossover(x, y)


# They work as expected.

# Finally, let's run GO:

# In[30]:


from heur_go import GeneticOptimization


# In[31]:


heur = GeneticOptimization(dj, maxeval=10000, N=10, M=30, Tsel1=0.5, Tsel2=0.1, 
                           mutation=cauchy, crossover=UniformMultipoint(1))
res = heur.search()
print(res['best_x'])
print(res['best_y'])
print(res['neval'])


# # _h_-means
# 
# Heuristic cluster analysis inspired by [_k_-means](https://en.wikipedia.org/wiki/K-means_clustering). Another demonstration of how continuous heuristics can be used — example of an application.

# In[32]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[33]:


from objfun_hmeans import HMeans


# In[34]:


of = HMeans()  


# In[35]:


print('f* = {}'.format(of.fstar))


# In[36]:


# plot the data points
X = of.X
ax = plt.scatter(x=X[:, 0], y=X[:, 1]);


# In[37]:


print('a = {}'.format(of.a))
print('b = {}'.format(of.b))


# **Bounds are repeated for each centroid, that will be tuned by the heuristic.**

# In[38]:


# some random evaluations
for i in range(10):
    x = of.generate_point()
    print('f({}) = {}'.format(x, of.evaluate(x)))


# In[39]:


# we can get cluster labels (for a random solution)
labels = of.get_cluster_labels(x)
print(labels)


# In[40]:


# auxiliary routine
def visualize_solution(x, of):
    labels = of.get_cluster_labels(x)
    X = of.X
    ax = plt.scatter(x=X[:, 0], y=X[:, 1], c=labels)


# In[41]:


# visualization of a random solution
visualize_solution(x, of);


# ## Optimization demonstration

# In[42]:


from heur_aux import MirrorCorrection, CauchyMutation
from heur_fsa import FastSimulatedAnnealing


# In[43]:


heur = FastSimulatedAnnealing(of, maxeval=10000, T0=10, n0=10, alpha=2, 
                              mutation=CauchyMutation(r=0.1, correction=MirrorCorrection(of)))
res = heur.search()
print('x_best = {}'.format(res['best_x']))
print('y_best = {}'.format(res['best_y']))
print('neval = {}'.format(res['neval']))


# In[44]:


visualize_solution(res['best_x'], of)


# # Assignments
# 
# * Re. _h_-means
#   * Improve the implementation of this objective function (especially the random solution generator)
#   * Tune heuristics for this objective function
#   * Tune this objective function, e.g. by penalization for smaller number of clusters than $h$ (and make sure you understand why this is possible)
#   * Compare heuristic approach to the original _k_-means
# * Tune heuristics on other continuous [benchmark functions](http://www.geatbx.com/docu/fcnindex-01.html)
