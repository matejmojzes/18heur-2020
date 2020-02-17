
# coding: utf-8

# # February 18 2020

# ## 1. Introduction
# 
# ### About me
# 
# **Matej Mojzeš**
# 
# * Ph.D. in integer optimization heuristics at the Deptartment of Software Engineering, FNSPE CTU in Prague
# * I also work(ed) as Head of Data Science at [Jumpshot](https://www.jumpshot.com)
# * More on my [LinkedIn profile](http://cz.linkedin.com/in/matejmojzes)

# ### About this course
# 
# We will experiment with "black-box" optimization of various **objective functions** (e.g. Travelling Salesman Problem, Artificial Neural Network weight optimization, [benchmark function optimization](http://www.geatbx.com/docu/fcnindex-01.html)) using different kinds of **heuristics** (e.g. Simulated Annealing, Genetic Optimization, Differential Evolution).
# 
# Recommended tools for this course are **Python & Jupyter notebook**:
# * Python: [reasons why](https://www.stat.washington.edu/~hoytak/blog/whypython.html), Jupyter notebook: [reasons why](http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261),
# * download & install from [here](https://www.continuum.io/downloads).
# 
# All **resources** will be on [GitHub](https://github.com/matejmojzes/18heur-2020). Any meaningful pull requests will be taken into account during your evaluation.

# ### About _zápočet_
# 
# I will enourage and help you to work on your own projects as long as:
# 
# * they are objective functions that can be optimized by heuristics and/or
# * they are heuristics, of course :-)
# 
# In the worst case, you should test and evaluate objective functions and heuristics that will be presented during our classes, but on a larger scale. It is completely up to you to mix your own creativity with large-scale experiments.
# 
# Your goal is to:
# 
# * deliver your work in form of a **research paper** with source code attached (ideally as a Jupyter notebook),
# * be patient and work hard.
# 
# <img src="img/journey_to_greatness.jpg">
# 
# _Please note that this picture contains nice example of heuristic optimization :-)_

# #### Notes
# 
# * Subscribe to the [mailing list](https://forms.gle/TuXwHgm9dTnamL9PA).
# * **DO NOT** try to begin working on your project in the last week of the school year (or before your final exams), chances are that:
#   * you will not have enough time to deliver a project of acceptable quality and/or
#   * I will be out of the office and off-line.
# * Last, but not least, nota bene:
# 
# <img src="img/learning.jpeg">
# 
# Therefore, I offer you the possibility to actively participate in a final "Show and tell" lecture, and I will improve your mark based on the successful presentation of your project

# ## 2. First assignment
# 
# <img src="img/airship.png">
# 
# 1. Imagine that you are on an air ship flying over discrete, 1-dimensional, terrain: $[0; 799]$.
# 2. Your task is to find the highest peak. You now its height ($y=100$), but not its location ($x$). It is hidden by the clouds.
# 3. You can measure terrain altitude only by planting paratroops - each will report his altitude after the jump.
# 4. You have only 100 paratroops on board, but otherwise you are not limited by any other constraints (air ship have unlimited fuel, there is no wind, etc...)
# 5. Your task: find the $x$ coordinate of the highest peak.
# 
# 
# * Simulate this experiment 1000 times.
# * When trying to find the highest peak, remember 1.) the highest altitude reached, 2.) number of paratroops planted before hitting the highest peak (you will compute some basic statistics based on these two vectors)

# In[1]:

# example of objective function implementation in Python

def evaluate(x):
    px = np.array([0,  50, 100, 300, 400, 700, 799], dtype=int)
    py = np.array([0, 100,   0,   0,  25,   0,  50], dtype=int)
    xx = np.arange(0, 800)
    yy = np.interp(xx, px, py)
    return -yy[x]  # negative altitude, becase we are minimizing (to be consistent with other obj. functions)


# ## 3. Results analysis
# 
# Let us compare two simple (but sometimes surprisingly efficient) heuristics:
# 
# 1. **Random Shooting (RS)**: evaluates random solutions until optimum is found or maximum number of evaluations is exhausted.
# 2. **Shoot & Go (SG)**: also known as _Random-restart hill climbing_, same as RS, but each random solution is iteratively improved using its ring neighbourhood if possible.
# 
# 
# #### 2nd question
# 
# > In how many experiments did you find the highest peak (x=50, y=100)?
# 
# * **RS** ~ $120/1000 = 12\%$ (we could expect $1/800\cdot100=0.125$)
# * **SG** ~ $180/1000 = 18\%$
# 
# 
# #### 3rd question
# 
# > When you found the highest peak (x=50, y=100), what was the median of number of paratroops used to find the peak?
# 
# * **RS** ~ $46$
# * **SG** ~ $55$
# 
# #### 4th question
# 
# > What was the median of the highest altitude found at the end of each experiment?
# 
# * **RS** ~ $94$
# * **SG** ~ $24.5$
# 
# #### Which of the two methods is better?
# 
# The answer depends on you, actually. Two basic criteria:
# * number of evaluations normalized by reliability: **Shoot & Go** ($55/0.18 = 305 < 46/0.12 = 383$)
# * distance from optimum: **Random Shooting** ($94 > 24.5$)
# 

# ## 4. Conclusions
# 
# When optimizing an objective function using heuristics:
# 
# * We know our domain (in this case, the 1-D terrain), but we suppose that we do not explicitly know the _function_ we are optimizing (in here: altitude – it's hidden by clouds)
# * We can only evaluate the function in some finite number of points (here: we are have only 100 paratroops at our disposal)
# * We typically know what is the objective function value we want to reach (here: $y=100$)
