import numpy.linalg as npla
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm_notebook


class BaseMethod(object):
    def __init__(self, oracle, x_0, stopping_criteria, trace):
        self.oracle = oracle
        self.x_0 = x_0.copy()
        self.trace = trace
        if stopping_criteria == 'grad_rel':
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == 'grad_abs':
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == 'func_abs':
            self.stopping_criteria = self.stopping_criteria_func_absolute
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        else:
            raise ValueError('Unknown stopping criteria type: "{}"'\
                             .format(stopping_criteria))
    
    def run(self, max_iter=10, max_time=1200):
        if not hasattr(self, 'hist'):
            self.hist = defaultdict(list)
        if not hasattr(self, 'time'):
            self.time = 0.
        
        self._absolute_time = datetime.now()
        try:
            for iter_count in range(max_iter):
                if self.time > max_time:
                    break
                if self.trace:
                    self._update_history()
                self.step()

                if self.stopping_criteria():
                    break
        except KeyboardInterrupt:
            print('Run interrupted at iter #{}'.format(iter_count))
        
        self.hist['x_star'] = self.x_k.copy()
    
    def _update_history(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now
        self.hist['func'].append(self.oracle.func(self.x_k))
        self.hist['time'].append(self.time)
        if not hasattr(self, 'grad_k'):
            self.grad_k = self.oracle.grad(self.x_k)
        self.hist['grad_norm'].append(npla.norm(self.grad_k))

    def step(self):
        raise NotImplementedError('step() not implemented!')
    
    def stopping_criteria_grad_relative(self):
        return npla.norm(self.grad_k)**2 <= self.tolerance * self.grad_norm_0**2

    def stopping_criteria_grad_absolute(self):
        return npla.norm(self.grad_k)**2 <= self.tolerance
    
    def stopping_criteria_func_absolute(self):
        return self.oracle.func(self.x_k) < self.tolerance
    
    def stopping_criteria_none(self):
        return False
