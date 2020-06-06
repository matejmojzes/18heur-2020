import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os.path


class ExperimentPerformer(object):

    """
    Allows experimenting with different kinds of heuristics at ones.
    """

    def __init__(self, log_file='experiments.log'):
        self.log_file = log_file
        self.run = 0
        self.all_results = pd.DataFrame()

    def get_all_results(self):
        return self.all_results

    def clear_results(self):
        self.all_results = pd.DataFrame()

    def load_history(self):
        self.all_results = pd.read_csv(self.log_file)
        return self.all_results

    def next_run(self):
        self.run += 1
        return self.run

    def save_results(self, exp_data_frame):
        exp_data_frame.to_csv(self.log_file, mode='a', header=(not os.path.isfile(self.log_file)))
        self.all_results = pd.concat([self.all_results, exp_data_frame], axis=0)

    def experiment(self, heur, num_runs):
        if not isinstance(heur, list):
            heur = [heur]
        exp_results = pd.DataFrame()
        for h in tqdm(heur, 'Experiment progress'):
            h_results = []
            for run in range(num_runs):
                result = h.search()
                result['run'] = self.next_run()
                result['of'] = h.of.get_name()
                result['heur'] = h.get_specs()
                result['maxeval'] = h.maxeval
                h_results.append(result)
                h.clear()
            h_data_frame = pd.DataFrame(h_results, columns=['of', 'heur', 'run', 'best_x', 'best_y', 'maxeval', 'neval'])
            self.save_results(h_data_frame)
            exp_results = pd.concat([exp_results, h_data_frame], axis=0)
        return exp_results

    def rel(self, x):
        return len([n for n in x if n < np.inf]) / len(x)

    def mne(self, x):
        values = [n for n in x if n < np.inf]
        if len(values):
            return np.mean(values)
        return np.inf

    def feo(self, x):
        rel = self.rel(x)
        if rel is 0:
            return np.inf
        mne = self.mne(x)
        if mne is np.inf:
            return np.inf
        return self.mne(x) / self.rel(x)

    def get_stats(self, index=None):
        if index is None:
            index = ['of', 'heur', 'maxeval']

        stats = self.all_results.pivot_table(
            index=index,
            values=['neval'],
            aggfunc=(self.rel, self.mne, self.feo)
        )['neval']
        stats = stats.reset_index()
        return stats.sort_values(by='rel', ascending=False)
