from heur import Heuristic, StopCriterion
import numpy as np


class DifferentialEvolution(Heuristic):

    def __init__(self, of, maxeval, N, CR, F):
        Heuristic.__init__(self, of, maxeval)
        assert N >= 4, 'N should be at least equal to 4'
        self.N = N  # Population size
        self.n = np.size(of.a)
        self.CR = CR  # Crossover probability
        assert 0 <= F <= 2, 'F should be from [0; 2]'
        self.F = F  # Differential weight
        self.name = 'DE'

    def get_specs(self):
        return self.get_name() + '_N={}_CR={}_F={}'.format(self.N, self.CR, self.F)

    def search(self):
        try:
            # Initialization
            n = np.size(self.of.a)
            pop_X = np.zeros([self.N, n], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population fitness (objective) function values
            for i in np.arange(self.N):
                x = self.of.generate_point()
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # Evolution iteration
            while True:
                for i in range(self.N):
                    x = pop_X[i]
                    agents = np.random.choice(np.delete(np.arange(self.N), i), 3, replace=False)  # selected 3 agents
                    a, b, c = pop_X[agents[0]], pop_X[agents[1]], pop_X[agents[2]]
                    R = np.random.randint(low=0, high=self.n)
                    y = [a[j] + self.F * (b[j] - c[j]) if np.random.rand() < self.CR or j == R else x[j] for j in
                         range(n)]
                    f_y = self.evaluate(y)
                    if f_y < pop_f[i]:
                        pop_X[i] = y
                        pop_f[i] = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise
