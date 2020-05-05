#!/usr/bin/env python
# coding: utf-8

# # Differential Evolution
# 
# For detailed description refer to [Wikipedia article](https://en.wikipedia.org/wiki/Differential_evolution).

# #### Set up IPython notebook environment first...

# In[1]:


# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().run_line_magic('pwd', '')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Demo

# In[2]:


from objfun_dejong1 import DeJong1
from heur_de import DifferentialEvolution


# In[3]:


dj = DeJong1(n=3, eps=0.01)


# In[4]:


de = DifferentialEvolution(of=dj, maxeval=1000, N=10, CR=0.5, F=1)
res = de.search()
print('x_best = {}'.format(res['best_x']))
print('y_best = {}'.format(res['best_y']))
print('neval = {}'.format(res['neval']))


# # Excercises
# 
# * Study the new algorithm performance
# * Experiment with alternative approaches to mutation (`DE/best/x` etc, see e.g. [this SO question](http://stackoverflow.com/questions/20393102/all-versions-of-differential-evolution-algorithm) for reference)
