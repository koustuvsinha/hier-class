## Model utils to load and save models
import torch
import torch.nn as nn
import os
from os.path import dirname, abspath
import json
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def create_save_dir(exp_name):
    # Create directory if not present in `saved` in the experiment name
    # going up 3 levels
    base_dir = str(os.path.dirname(os.path.realpath(__file__)).split('codes')[0])
    save_path_base = os.path.join(base_dir, 'saved', exp_name)
    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)
    return save_path_base

def save_model(model, epoch=0, step=0, exp_name='', params=None):
    """ Save model, model params
    Check if model params serializable type then save
    """

    save_path_base = create_save_dir(exp_name)
    save_path = save_path_base + '/' + \
                params['save_name'].format(epoch, step)
    logging.info("Saving model in {}".format(save_path))
    if hasattr(model, "save_state_dict"):
        model.save_state_dict(save_path)
    else:
        torch.save(
            model.state_dict(),
            save_path
        )
    logging.info("Model saved, now saving parameters")
    # nix params which are not json serializable
    to_save = {}
    for key, val in params.items():
        if is_jsonable(val):
            to_save[key] = val
    json.dump(to_save, open(save_path_base + '/parameters.json', 'w'))
    logging.info("Saved model and params")

def get_mlp(input_dim, output_dim, num_layers=2, dropout=0):
    network_list = []
    assert num_layers > 0
    if num_layers > 1:
        for _ in range(num_layers-1):
            network_list.append(nn.Linear(input_dim, input_dim))
            network_list.append(nn.ReLU())
            network_list.append(nn.Dropout(dropout))
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )

class SLTR():
    """
    Slanted Triangular Learning Rate
    paper: https://arxiv.org/pdf/1801.06146.pdf
    """
    def __init__(self, epochs, batch_size, num_train,
                 cut_frac=0.1, ratio=1000, lr_max = 0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_train = num_train
        self.updates = num_train // batch_size
        self.T = self.updates * epochs
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr_max = lr_max
        self.cut = np.floor(self.T * cut_frac)
        self.t = 0

    def step(self, optimizer):
        self.t += 1
        if self.t < self.cut:
            p = self.t / self.cut
        else:
            p =  1 - (self.t - self.cut) / (self.cut * (1/self.cut_frac - 1))
        new_lr = (self.lr_max * (1 + p*(self.ratio - 1))) / self.ratio
        if self.t % self.updates == 0:
            logging.info("LR : {}".format(new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return optimizer


