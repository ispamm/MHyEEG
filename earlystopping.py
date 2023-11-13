"""Taken from https://github.com/Bjarten/early-stopping-pytorch"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='max'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print      
            mode (str): 'min' to save model when metric decreases (e.g. loss), 'max' when it increases (e.g. accuracy).
                            Default: 'max'        
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        
    def __call__(self, metric, model):
        if self.mode == 'min':
            #Loss saved
            score = -metric
        else:
            score = metric

        if self.best_score is None:  # initial step
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:  # if not improving (i.e. not growing)
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # if improved (i.e. grown)
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when metric imroves.'''
        if self.verbose: 
            if metric > 0:
                self.trace_func(f'Validation auc increased ({self.val_metric_min:.6f} --> {metric:.6f}).  Saving model ...')
            else:  # case a loss is given
                self.trace_func(f'Validation auc increased ({-self.val_metric_min:.6f} --> {-metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_min = metric