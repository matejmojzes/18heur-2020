from heur_aux import is_integer
from heur import Heuristic, StopCriterion
import numpy as np


class Crossover:

    """
    Baseline crossover  - randomly chooses "genes" from parents
    """

    def __init__(self):
        self.name = 'mix'

    def crossover(self, x, y):
        z = np.array([x[i] if np.random.uniform() < 0.5 else y[i] for i in np.arange(x.size)], dtype=x.dtype)
        return z

    def get_name(self):
        return self.name


class UniformMultipoint(Crossover):

    """
    Uniform n-point crossover
    """

    def __init__(self, n):
        self.n = n  # number of crossover points
        self.name = 'uni'

    def crossover(self, x, y):
        co_n = self.n + 1
        n = np.size(x)
        z = x*0
        k = 0
        p = np.ceil(n/co_n).astype(int)
        for i in np.arange(1, co_n+1):
            ix_from = k
            ix_to = np.minimum(k+p, n)
            z[ix_from:ix_to] = x[ix_from:ix_to] if np.mod(i, 2) == 1 else y[ix_from:ix_to]
            k += p
        return z


class RandomCombination(Crossover):

    """
    Randomly combines parents
    """

    def __init__(self):
        self.name = 'rnd'

    def crossover(self, x, y):

        if is_integer(x):
            z = np.array([np.random.randint(np.min([x[i], y[i]]), np.max([x[i], y[i]]) + 1) for i in np.arange(x.size)],
                         dtype=x.dtype)
        else:
            z = np.array([np.random.uniform(np.min([x[i], y[i]]), np.max([x[i], y[i]])) for i in np.arange(x.size)],
                         dtype=x.dtype)
        return z


class GeneticOptimization(Heuristic):

    def __init__(self, of, maxeval, N, M, Tsel1, Tsel2, mutation, crossover):
        Heuristic.__init__(self, of, maxeval)

        assert M > N, 'M should be larger than N'
        self.N = N  # population size
        self.M = M  # working population size
        self.Tsel1 = Tsel1  # first selection temperature
        self.Tsel2 = Tsel2  # second selection temperature
        self.mutation = mutation
        self.crossover = crossover

        self.name = 'GO{}{}'.format(self.crossover.get_name(), mutation.get_name())

    def get_specs(self):
        return '{}_N={}_M={}_T1={}_T2={}'.format(self.get_name(), self.N, self.M, self.Tsel1, self.Tsel2)

    @staticmethod
    def sort_pop(pop_x, pop_f):
        ixs = np.argsort(pop_f)
        pop_x = pop_x[ixs]
        pop_f = pop_f[ixs]
        return [pop_x, pop_f]

    @staticmethod
    def rank_select(temp, n_max):
        u = np.random.uniform(low=0.0, high=1.0, size=1)
        ix = np.minimum(np.ceil(-temp*np.log(u)), n_max)-1
        return ix.astype(int)

    def search(self):
        try:
            # Initialization:
            pop_X = np.zeros([self.N, np.size(self.of.a)], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population fitness (objective) function values
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
                work_pop_X = np.zeros([self.M, np.size(self.of.a)], dtype=self.of.a.dtype)
                work_pop_f = np.zeros(self.M)
                for i in np.arange(self.M):
                    parent_a_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # select first parent
                    parent_b_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # 2nd --//-- (not unique!)
                    par_a = pop_X[parent_a_ix, :][0]
                    par_b = pop_X[parent_b_ix, :][0]
                    z = self.crossover.crossover(par_a, par_b)
                    z_mut = self.mutation.mutate(z)
                    work_pop_X[i, :] = z_mut
                    work_pop_f[i] = self.evaluate(z_mut)

                # 2.) sort working population according to fitness function
                [work_pop_X, work_pop_f] = self.sort_pop(work_pop_X, work_pop_f)

                # 3.) select the new population
                ixs_not_selected = np.ones(self.M, dtype=bool)  # this mask will prevent us from selecting duplicates
                for i in np.arange(self.N):
                    sel_ix = self.rank_select(temp=self.Tsel2, n_max=np.sum(ixs_not_selected))
                    pop_X[i, :] = work_pop_X[ixs_not_selected][sel_ix, :]
                    pop_f[i] = work_pop_f[ixs_not_selected][sel_ix]
                    ixs_not_selected[sel_ix] = False

                # 4.) sort according to fitness function
                [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

        except StopCriterion:
            return self.report_end()
        except:
            raise
