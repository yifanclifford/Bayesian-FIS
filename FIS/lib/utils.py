import ctypes

import numpy as np
import pytrec_eval
import torch

from pathlib import Path
import datetime


def log_loss_logits(target, input):
    logloss = np.clip(input, 0, None) - input * target + np.log1p(np.exp(-np.abs(input)))
    return np.sum(logloss) / len(input)


def mkdir_safe(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_lib(base_path):
    lib_file = 'cpp/release/lib.so'
    lib = ctypes.cdll.LoadLibrary(lib_file)
    lib.init(ctypes.create_string_buffer(base_path.encode(), len(base_path) * 2))
    lib.sample_negative.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64
    ]
    lib.all_negative.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64
    ]
    lib.get_num_negative.argtypes = [ctypes.c_int64]
    return lib


def sample_negative(lib, user, num_negative):
    neg_items = np.zeros(num_negative, dtype=np.int64)
    neg_items_addr = neg_items.__array_interface__["data"][0]
    lib.sample_negative(neg_items_addr, user, num_negative)
    return neg_items


def read_split(x):
    return [int(i) for i in x.split(' ')]


def current_timestamp():
    current = datetime.datetime.now()
    timestamp = f'{current.year}{current.month}{current.day}{current.hour}{current.minute}{current.second}'
    return timestamp


class Evaluator:
    def __init__(self, metrics):
        self.result = None
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model_path, device, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model_path = model_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = np.Inf
        self.delta = delta
        self.device = device

    def reset(self):
        self.counter = 0
        self.val_score_max = np.Inf
        self.early_stop = False
        self.best_score = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decrease ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ...')
        model.cpu()
        torch.save(model.state_dict(), self.model_path)
        model.to(self.device)
        self.val_score_max = val_score
