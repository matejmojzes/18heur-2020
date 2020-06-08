from heur import Heuristic, StopCriterion
from heur_aux import Correction, CauchyMutation
import numpy as np


class DifferentialEvolution(Heuristic):

    def __init__(self, of, maxeval, N, CR, F):
        Heuristic.__init__(self, of, maxeval)
        assert N >= 4, 'N should be at least equal to 4'
        self.N = int(N) # POPULATION SIZE
        self.CR = CR # CROSSOVER PROBABILITY
        assert 0 <= F <= 2, 'F should be from [0; 2]'
        self.F = F # DIFFERENTIAL WEIGHT
        self.mutation = CauchyMutation(r=0.75, correction=Correction(of))

    def search(self):
        try:
            # Initialization
            n = np.size(self.of.a) # size of solution
            # population solution vectors
            pop_X = np.zeros([self.N, n], dtype=self.of.a.dtype)
            # population fitness (objective) function values
            pop_f = np.zeros(self.N)
            for i in np.arange(self.N):
                x = self.of.generate_point()
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # Evolution iteration
            while True:
                for i in range(self.N):
                    x = pop_X[i]
                    # selected 3 agents - indexes
                    agents= np.random.choice(np.delete(np.arange(self.N), i), \
                                             3, replace=False)  
                    a, b, c=pop_X[agents[0]], pop_X[agents[1]],pop_X[agents[2]]
                    R = np.random.randint(low=0, high=self.N)
                    try:
                        y = [a[j] + self.F*(b[j]-c[j]) if np.random.rand() < \
                             self.CR or j == R else x[j] for j in range(n)]
                    except:
                        print("No update in Differenntial evolution")
                    y = np.array(y)
                    y = list(y.astype(int))                 
                    y = self.mutation.mutate(y)
                    f_y = self.evaluate(y)
                    if f_y < pop_f[i]:
                        pop_X[i] = y
                        pop_f[i] = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise