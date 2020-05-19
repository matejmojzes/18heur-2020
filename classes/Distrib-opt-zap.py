#!/usr/bin/env python
# coding: utf-8

# $\boldsymbol{x}\in\mathbb{R}^{d}$, kde d$\in\mathbb{N}$ je konečné, např. 100. Dál pro $\boldsymbol{x}$ platí 
# 
# $\sum_{k=1}^{d}x_{k}=1\wedge\left(\forall k\in\left\{ 1,2,\ldots,d\right\} \right)\left(\boldsymbol{x}_{k}>0\right)$.
# 
# Tato podmínka dává $\boldsymbol{x}$ formu distribuce. 
# 
# V mojí optimalizaci mám data $\left\{ \alpha\right\} _{i\in\mathbb{N}}\in\mathbb{R}^{d}$ bez dalších omezení. Optimalizuju funkcionál
# 
# $-\sum_{i}\boldsymbol{\alpha}_{i}\cdot\boldsymbol{x}$,
# 
# kde struktura $\alpha_{i,k}=\left\{ \begin{array}{cc}
# k\Delta-\Delta_{i,0}, & \text{ pro }k<m_{i},\\
# 0, & \text{ jinak}.
# \end{array}\right.$.
# 
# To znamená, že $\boldsymbol{\alpha}_{i}$ závisí pouze na dvou parametrech $\Delta_{i}\in\mathbb{R}$ a $0\leq m_{i}\leq d$.

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


# In[3]:


import objfun_distribution_opt as dopt


# In[4]:


alphas = [{"r": 15, "m": 10},{"r": 12, "m": 9.5},{"r": 12, "m": 9.5},{"r": 12, "m": 9.5},{"r": 12, "m": 9.5}]
delta = 1
d = 15
problem = dopt.VolumeDistribution(d, delta, alphas)

x = problem.generate_point()
# print(x)
# x = np.reshape([0 if i != 9 else 1 for i in range(0,d)],(1, d))
# print(x)
print(problem.alphas)

print(problem.evaluate(x))


# - penalizace
# - jako $f^{*}$ nastavit nějaké dobré řešení

# ### Heuristics application

# In[5]:


alphas = [{"r": 15, "m": 10},{"r": 5, "m": 5},{"r": 6, "m": 6},{"r": 5, "m": 5},{"r": 7, "m": 7}, {"r": 20, "m": 1},
          {"r": 0, "m": 14},{"r": 0, "m": 14},{"r": 0, "m": 14},{"r": 0, "m": 14},{"r": 0, "m": 14}]
delta = 1
d = 15
prob_unlimited = dopt.VolumeDistribution(d, delta, alphas)

x_star = np.zeros(d)
s = 0
for a in alphas:
    x_star[int(a["m"] / delta)] += a["r"]
    s += a["r"]
x_star /= s

print(prob_unlimited.evaluate(x_star))
prob = dopt.VolumeDistribution(d, delta, alphas, prob_unlimited.evaluate(x_star))


# In[6]:


for i in range(100):
    _x = prob.generate_point()
    f = prob.evaluate(_x)
    if f < prob.get_fstar(): 
        print((_x, prob.evaluate(_x)))


# ## Random shooting test

# In[7]:


from heur_sg import ShootAndGo

for i in range(10):
    sg = ShootAndGo(prob, 1000, 0)
    sg.search()
    print(sg.best_y, sg.best_x, sg.neval)


# ### Genetic optimization
# when using the convex combination, the solution remains in the domain

# In[8]:


import heur_aux

normCorr = heur_aux.NormalizeCorrection(prob)
xx = np.reshape([1 for i in range(15)],15)
print(normCorr.correct(xx))

## vectors = np.reshape([[-1 if i == j-1 else (1 if i == j else 0) for i in range(15)]for j in range(1,15)], (14, 15)).T
## print(vectors)
## print(np.dot(vectors[:,1],  vectors[:,0]))
## q, r = np.linalg.qr(vectors, mode = 'complete')
## 
## print(np.dot(q.T,np.dot(q.T,xx)))
## print(np.linalg.lstsq(vectors, xx))
## og_proj = heur_aux.OGProjectionCorrection(prob, vectors)


# In[9]:


from heur_go import Crossover, UniformMultipoint, RandomCombination


# In[10]:


from heur_go import GeneticOptimization
from heur_fsa import FastSimulatedAnnealing


# In[11]:


from heur_aux import CauchyMutation, Mutation, NormalizeCorrection, ExtensionCorrection, MirrorCorrection, Correction


# In[12]:


cauchy = CauchyMutation(r=1, correction=NormalizeCorrection(prob))
cauchy.mutate(np.array([6.12, -4.38,  2.96]))


# In[13]:


heur = GeneticOptimization(prob, maxeval=10000, N=10, M=30, Tsel1=0.5, Tsel2=0.1, 
                           mutation=cauchy, crossover=UniformMultipoint(1))
res = heur.search()
res


# In[14]:


heur = GeneticOptimization(prob, maxeval=10000, N=10, M=30, Tsel1=0.5, Tsel2=0.1, 
                           mutation=cauchy, crossover=RandomCombination())
res = heur.search()
res


# In[15]:


sum(res['best_x'])


# In[16]:


heur = GeneticOptimization( prob_unlimited, 
                            maxeval=10000,
                            N=10,
                            M=30,
                            Tsel1=0.5,
                            Tsel2=0.1, 
                            mutation=cauchy, crossover=UniformMultipoint(1) )
res = heur.search()
res


# In[17]:


heur = GeneticOptimization(prob_unlimited, maxeval=10000, N=10, M=30, Tsel1=0.5, Tsel2=0.1, 
                           mutation=cauchy, crossover=RandomCombination())
res = heur.search()
res


# In[18]:


import matplotlib.pyplot as plt

plt.plot(res['best_x'])


# ### Now, let us move to more complex setup of the problem

# In[19]:


np.random.seed(123545)
alphas = [{"r": int(np.random.gamma(2, 5)), "m": int(np.random.gamma(1, 9))} for i in range(300)]

for alpha in alphas:
    if alpha['m'] >= 25:
        alpha['m'] = 25
        alpha['r'] = 0
    if alpha['r'] - alpha['m'] > 10:
        alpha['r'] = 10 + alpha['m']

#alphas = [{"r": i *2, "m": i * 2} for i in range(10)] 
#alphas = [{"r": 3, "m": 12}, {"r": 2, "m": 15}, {"r": 0, "m": 20} ] 
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_ylabel('r')
ax1.set_xlabel('m')
ax1.set_title('data')
line, = ax1.plot([alpha['m'] for alpha in alphas], [alpha['r'] for alpha in alphas], 'bo')

plt.show()


# In[20]:


delta = 1
d = int(max([alpha['m'] for alpha in alphas])/delta) + 1
prob_unlimited = dopt.VolumeDistribution(d, delta, alphas)
prob_pen_unlimited = dopt.VolumeDistributionPenalization(d, delta, alphas, -np.inf, 1e-2)
x_star = np.zeros(d)
s = 0
for a in alphas:
    x_star[int(a["m"] / delta)] += a["r"]
    s += a["r"]
x_star /= s

plt.figure()
plt.plot(np.cumsum(x_star))
print(prob_unlimited.evaluate(x_star))

## Setup the problem of volume distribution fitness
prob = dopt.VolumeDistribution(d, delta, alphas, prob_unlimited.evaluate(x_star))
## Setup the problem with penalization
prob_pen = dopt.VolumeDistributionPenalization(d, delta, alphas, prob_unlimited.evaluate(x_star), 0)


# Problem analysis

# In[22]:


heur = GeneticOptimization( prob_unlimited, 
                            maxeval=10000,
                            N=50,
                            M=150,
                            Tsel1=0.5,
                            Tsel2=0.1, 
                            mutation=cauchy, crossover=UniformMultipoint(1) )
res = heur.search()
print(res['best_y'])
plt.plot(np.cumsum(res['best_x']))


# In[23]:


heur = GeneticOptimization( prob_unlimited, 
                            maxeval=10000,
                            N=10,
                            M=30,
                            Tsel1=0.5,
                            Tsel2=0.2, 
                            mutation=cauchy, crossover=RandomCombination() )
res = heur.search()
print(res['best_y'])
plt.plot(np.cumsum(res['best_x']))


# In[24]:


heur = FastSimulatedAnnealing( prob_unlimited,
                               maxeval=1000,
                               T0=1,
                               n0=1, 
                               alpha=2, 
                               mutation=cauchy )
res = heur.search()
print(res['best_y'])
plt.plot(np.cumsum(res['best_x']))


# ### utilization of problem with penalization

# In[81]:


NUM_RUNS = 100
resultsPen = pd.DataFrame()
runs = []
corrections = [ {"name": "correction", 'corr': Correction(prob_pen)},
                {"name": "extension", 'corr': ExtensionCorrection(prob_pen)},
                {"name": "mirror", 'corr': MirrorCorrection(prob_pen)} ]
for correction in corrections: 
    for Cr in [0.01, 0.1, 0.5, 1]:
        for p in [0.8, 0.1, 0.01, 0.005]:
            prob_pen_unlimited.penalization = p
            heur_name = 'GO_DOP(p:{},Cr:{},corr{})'.format(p, Cr, correction['name'])
            for i in tqdm(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
                run = GeneticOptimization( prob_pen_unlimited, 
                                            maxeval=1000,
                                            N=50,
                                            M=100,
                                            Tsel1=0.5,
                                            Tsel2=0.2, 
                                            mutation=CauchyMutation(r=Cr, correction=correction['corr']),
                                            crossover=UniformMultipoint(1) ).search()
                run['run'] = i
                run['heur'] = heur_name
                run['penalization'] = p
                run['correction'] = correction['name']
                run['Cr'] = Cr
                runs.append(run)

            res_df = pd.DataFrame(runs,
                                  columns=['heur',
                                           'penalization', 
                                           'correction', 
                                           'Cr',
                                           'run',
                                           'best_x',
                                           'best_y',
                                           'neval'] )
            resultsPen = pd.concat([results, res_df], axis=0)


# In[83]:


def avg_sum(x):
    return np.mean(np.sum(x))
def admissibility_rate(x):
    return len([n for n in x if np.sum(n) == 1])/len(x)

results_pivot = resultsPen.pivot_table(
    index=['penalization', 'heur'],
    values=['best_x'],
    aggfunc=(avg_sum, admissibility_rate)
)['best_x']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='admissibility_rate')


# In[98]:


results_pivot = resultsPen.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='rel')


# ### Comparison

# In[93]:


## Setup the best found value 
prob.fstar = -50

NUM_RUNS = 100
maxeval = 1000


# In[30]:


import pandas as pd
from tqdm.notebook import tqdm


# In[94]:


results = pd.DataFrame()
runs = []
for setup in [{'N': 50, 'M': 150, 'multipoint': 1, 'T1': 0.5, 'T2': 0.1, 'Cr': 1,},
              {'N': 50, 'M': 100, 'multipoint': 1, 'T1': 0.5, 'T2': 0.1, 'Cr': 1,},
              {'N': 50, 'M': 100, 'multipoint': 1, 'T1': 0.5, 'T2': 0.2, 'Cr': 1,},
              {'N': 50, 'M': 100, 'multipoint': 1, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.1,},
              {'N': 50, 'M': 100, 'multipoint': 1, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.2,},
              {'N': 50, 'M': 100, 'multipoint': 1, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.01,},
              {'N': 50, 'M': 100, 'multipoint': 2, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.1,},
              {'N': 50, 'M': 100, 'multipoint': 4, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.1,},
              {'N': 50, 'M': 100, 'multipoint': 6, 'T1': 0.5, 'T2': 0.1, 'Cr': 0.1,}
             ]:
    heur_name = 'GO_(N:{},M:{},T1:{},T2:{},Cr:{},mult:{})'.format( setup['N'],
                                                                   setup['M'],
                                                                   setup['T1'],
                                                                   setup['T2'],
                                                                   setup['Cr'],
                                                                   setup['multipoint'] )
    for i in tqdm(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = GeneticOptimization( prob, 
                                   maxeval=maxeval,
                                   N=setup['N'],
                                   M=setup['M'],
                                   Tsel1=setup['T1'],
                                   Tsel2=setup['T2'], 
                                   mutation=CauchyMutation(r = setup['Cr'], correction =NormalizeCorrection(prob)),
                                   crossover=UniformMultipoint(setup['multipoint']) ).search()
        run['run'] = i
        run['heur'] = heur_name
        runs.append(run)
    
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)


# In[90]:


def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[95]:


results_pivot = results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='rel')


# In[92]:


results


# In[96]:


results = pd.DataFrame()
runs = []
heur_name = 'RS_{}'.format(maxeval)
for i in tqdm(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
    run = ShootAndGo(prob, maxeval, 0).search()
    run['run'] = i
    run['heur'] = heur_name
    runs.append(run)
    
res_df = pd.DataFrame(runs, columns=['heur', 'run', 'best_x', 'best_y', 'neval'])
results = pd.concat([results, res_df], axis=0)


# In[97]:


results_pivot = results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='rel')


# In[37]:


results


# In[ ]:




