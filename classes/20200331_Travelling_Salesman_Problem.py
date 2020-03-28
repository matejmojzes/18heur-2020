#!/usr/bin/env python
# coding: utf-8

# # Travelling Salesman Problem (TSP) on a rectangular grid 
# 
# ### Objective function description
# 
# * Cities placed on a rectangular grid in $\mathbb{R}^n$, dimension given by $A, B \in \mathbb{N}$
# * Assuming Euclidean distance the optimal tour has the length  
#   * $A \cdot B + \sqrt{2} - 1$ if $A$ and $B$ are even numbers
#   * $A \cdot B$ otherwise
# 
# ### How to find optimal tour using heuristics?
# 
# Our success heavily depends on efficient solution encoding. Rather extreme, binary, representation would result in $2^{n^2}$ states. Let us consider an encoding using $(n-1)!$ states "only", as demonstrated on the following example with $A=3$,  $B=2$ grid:
#   
# <img src="img/tsp_example.png">
# 
# Tour (in green) starts in city 0 and ends 1, proceeding via 2, 4, 5, 3. This corresponds to encoded solution $(1, 2, 2, 1, 0)$, i.e. indices of the selected remaining cities in each step.
# 
# Notes:
# 
# * $n = A \cdot B$
# * $\mathbf{a} = (0, 0, \ldots, 0)$ 
# * $\mathbf{b} = (n-2, n-3, \ldots, 0)$
# * $f^*$ quals to 
#   * $A \cdot B + \sqrt{2} - 1$ if both $A$ and $B$ are even numbers
#   * $A \cdot B$ otherwise
# * **For serious TSP optimization you should use much more sophisticated approaches, e.g. the [Concorde](https://en.wikipedia.org/wiki/Concorde_TSP_Solver) algorithm**

# # Example Implementation
# 
# You can find it in `src/objfun_tsp.py`, class `TSPGrid`.
# 
# Real-world demonstration follows:

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


# Import extrenal librarires
import numpy as np

# Import our code
from heur_sg import ShootAndGo
from objfun_tsp import TSPGrid  # <-- our implementation of TSP


# ### ``TSPGrid(3, 2)`` demonstration

# In[3]:


# initialization
tsp = TSPGrid(3, 2)


# In[4]:


# random point generation
x = tsp.generate_point()
print(x)


# In[5]:


# decode this solution (into list of visited cities)
cx = tsp.decode(x)
print(cx)


# In[6]:


# what is the cost of such tour?
of_val = tsp.evaluate(x)
print(of_val)


# In[7]:


# what is the cost of our example tour?
of_val = tsp.evaluate([1, 2, 2, 1, 0])
print(of_val)


# In[8]:


# neighbourhood of x:
N = tsp.get_neighborhood(x, 1)
print(N)


# In[9]:


# decoded neighbours and their objective function values
for xn in N:
    print('{} ({}) -> {:.4f}'.format(xn, tsp.decode(xn), tsp.evaluate(xn)))


# **Carefully** mind the difference between encoded solution vector vs decoded city tour and meaning of such neighbourhood.

# ### TSP optimization using Random Shooting ($\mathrm{SG}_{0}$)

# In[10]:


heur = ShootAndGo(tsp, maxeval=1000, hmax=0)
print(heur.search())


# # Assignments:
# 
# 1. Try to find a better performing heuristic (to test TSP implementation on your own).
# 2. Can you improve heuristic performance using any 
#    1. **better random point generator**?
#    2. **better neighbourhood generator**?
# 
# Use performance measures of your choice, e.g. $FEO$.
