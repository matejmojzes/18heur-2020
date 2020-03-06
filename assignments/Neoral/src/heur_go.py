from heur_aux import is_integer, CauchyMutation, Correction
from heur import Heuristic, StopCriterion
import numpy as np

class Crossover:
    """
    Baseline crossover  - randomly chooses "genes" from parents
    """

    def __init__(self):
        pass

    # this is basic
    def crossover(self, x, y):
        z = np.array([x[i] if np.random.uniform() < 0.5 else y[i] \
                      for i in np.arange(x.size)], dtype=x.dtype)
        return z


class UniformMultipoint(Crossover):
    """
    Uniform n-point crossover
    """

    def __init__(self, n):
        self.n = n  # number of crossover points

    def crossover(self, x, y): # we take two parents
        co_n = self.n + 1 # number of crossover points plus beginning
        n = np.size(x) # what is the length of parent?
        z = x*0 # prepare new vector z for our offspring
        k = 0 # we start at index 0
        p = np.ceil(n/co_n).astype(int) # what is the length of shift?
        for i in np.arange(1, co_n+1): # we move by shifts
            ix_from = k # were we start
            ix_to = np.minimum(k+p, n) # we finish p elements right or at end
            z[ix_from:ix_to] = x[ix_from:ix_to] if np.mod(i, 2) == 1 \
            else y[ix_from:ix_to] # if odd, copy part of x to z, otherwise y
            k += p # we move our starting point by our shift
        return z


class RandomCombination(Crossover):
    """
    Randomly combines parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):

        if is_integer(x):
            z = np.array([np.random.randint(np.min([x[i], y[i]]),\
                        np.max([x[i], y[i]]) + 1) \
                        for i in np.arange(x.size)], dtype=x.dtype)
        else:
            z = np.array([np.random.uniform(np.min([x[i], y[i]]), \
                        np.max([x[i], y[i]]) + 1) \
                        for i in np.arange(x.size)], dtype=x.dtype)
        return z


class GeneticOptimization(Heuristic):

    def __init__(self, of, maxeval, N, M, Tsel1, Tsel2):
        Heuristic.__init__(self, of, maxeval)

        assert M > N, 'M should be larger than N'
        self.N = int(N)  # population size
        self.M = int(M)  # working population size
        self.Tsel1 = Tsel1  # first selection temperature
        self.Tsel2 = Tsel2  # second selection temperature
        self.mutation = CauchyMutation(r=0.75, correction=Correction(of))
        self.crossover = UniformMultipoint(1)

    @staticmethod
    def sort_pop(pop_x, pop_f): # population x and and fitness for each member
        ixs = np.argsort(pop_f) # vector of members with least fitness 
        pop_x = pop_x[ixs] # rearrange population x by their fitness
        pop_f = pop_f[ixs] # now rearrange fitness to corespond for population
        return [pop_x, pop_f]

    @staticmethod
    def rank_select(temp, n_max): # temperature and population size
        u = np.random.uniform(low=0.0, high=1.0, size=1) # number between 0-1
        # at begining we choose as parent more widely and also those with
        # worse fitness, when temperature decreases, we take only the top ones
        ix = np.minimum(np.ceil(-temp*np.log(u)), n_max)-1
        return ix.astype(int)

    def search(self):
        try:
            # Initialization:
            # population solution vectors
            pop_X = np.zeros([self.N, np.size(self.of.a)],
                            dtype=self.of.a.dtype)
            # population fitness (objective) function values
            pop_f = np.zeros(self.N)  
            # a.) generate the population
            for i in np.arange(self.N):
                x = self.of.generate_point()
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # b.) sort according to fitness function
            [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

            # Evolution iteration
            while True:
                # 1.) generate the working population
                work_pop_X = np.zeros([self.M, np.size(self.of.a)], \
                                       dtype=self.of.a.dtype)
                work_pop_f = np.zeros(self.M) # fitness for that population
                for i in np.arange(self.M): # till size of population
                    # select first parent
                    parent_a_ix = self.rank_select(temp=self.Tsel1, \
                                                   n_max=self.N)
                    # 2nd --//-- (not unique!)
                    parent_b_ix=self.rank_select(temp=self.Tsel1, n_max=self.N)
                    # 0 because it is list of lists so [0] takes all vector
                    par_a = pop_X[parent_a_ix, :][0] # parent vector a
                    par_b = pop_X[parent_b_ix, :][0] # parent vector b
                    z = self.crossover.crossover(par_a, par_b) #randomly choose
                    z_mut = self.mutation.mutate(z)
                    work_pop_X[i, :] = z_mut
                    work_pop_f[i] = self.evaluate(z_mut)

                # 2.) sort working population according to fitness function
                [work_pop_X, work_pop_f] = self.sort_pop(work_pop_X, work_pop_f)

                # 3.) select the new population
                # this mask will prevent us from selecting duplicates
                ixs_not_selected = np.ones(self.M, dtype=bool)  
                for i in np.arange(self.N):
                    sel_ix = self.rank_select(temp=self.Tsel2, \
                                              n_max=np.sum(ixs_not_selected))
                    pop_X[i, :] = work_pop_X[ixs_not_selected][sel_ix, :]
                    pop_f[i] = work_pop_f[ixs_not_selected][sel_ix]
                    ixs_not_selected[sel_ix] = False

                # 4.) sort according to fitness function
                [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

        except StopCriterion:
            return self.report_end()
        except:
            raise
