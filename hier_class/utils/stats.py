# class to maintain statistics and logging utils

import numpy as np
from tensorboardX import SummaryWriter
import time
import json
import logging
import pdb
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from hier_class.utils import model_utils as mu


class Statistics():
    """
    Class to collect training and validation statistics.
    Also collect validation samples and attention for later inspection
    """
    def __init__(self, batch_size=0, max_levels=3, exp_name='', data=None, n_heads=[], level=-1):
        self.epoch = -1
        self.step = 0
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.batch_size = batch_size
        self.max_levels = max_levels
        self.exp_name = exp_name
        self.data = data
        self.n_heads = n_heads
        self.level = level
        self.writer = SummaryWriter(log_dir='../../logs/' + exp_name)
        self.output = {} # json file to store validation examples. should contain true and predicted labels (actual class names), validation examples, and generated attentions per epoch.
        self.output['train_indices'] = data.train_indices
        self.output['val_indices'] = data.test_indices

    def next(self):
        """Update the epoch and start timer"""
        self.epoch += 1
        self.step = 0
        self.calc_start = time.time()
        self.output[self.epoch] = {'attentions':[], 'val_indices':[], 'predictions':[]}
        save_path_base = mu.create_save_dir(self.exp_name)
        #json.dump(self.output, open(save_path_base + '/val_logs.txt','w'))
        logging.info("Epoch : {}".format(self.epoch))


    def update_train(self, train_loss, train_accuracy):
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.step +=1

    def update_validation(self, validation_loss, validation_accuracy, attn=None, src=None, preds=None):
        self.validation_loss.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)
        # TODO: store attentions (all layers)
        # TODO: convert src into words and store them in json
        """
        flat_attns = []
        for i,level_attn_var in enumerate(attn):
            comb_attn = level_attn_var.data.cpu().numpy()
            # separate the attentions by layer
            flat_attn = comb_attn.reshape(-1, self.n_heads[i], comb_attn.shape[2])
            flat_attns.append(flat_attn)
        flat_attns = np.hstack(flat_attns)
        # flatten
        src = [x for s in src for x in s]
        self.output[self.epoch]['attentions'].append(flat_attns.tolist())
        self.output[self.epoch]['val_indices'].append(src)
        preds_list = np.hstack([np.expand_dims(p.cpu().numpy(), axis=1) for p in preds]).tolist()
        self.output[self.epoch]['predictions'].append(preds_list)
        """


    def get_train_loss(self):
        return np.mean(self.train_loss)

    def get_valid_loss(self):
        return np.mean(self.validation_loss)

    def get_train_acc(self, level=0):
        train_acc = self.train_accuracy
        return np.mean([tr[level] for tr in train_acc])

    def get_valid_acc(self, level=0):
        valid_acc = self.validation_accuracy
        return np.mean([tr[level] for tr in valid_acc])

    def log_loss(self):
        time_taken = time.time() - self.calc_start
        logging.info('Time taken: {}'.format(time_taken * 1000))
        logging.info("After Epoch {}".format(self.epoch))
        logging.info("Train Loss : {}".format(self.get_train_loss()))
        self.writer.add_scalar('train_loss',self.get_train_loss(),self.epoch)
        logging.info("Validation Loss : {}".format(self.get_valid_loss()))
        self.writer.add_scalar('validation_loss', self.get_valid_loss(), self.epoch)
        for level in range(self.max_levels):
            logging.info("Train accuracy for level {} : {}".format(
                level, self.get_train_acc(level)))
            self.writer.add_scalar('train_acc_{}'.format(level),
                                   self.get_train_acc(level), self.epoch)
            logging.info("Validation accuracy for level {} : {}".format(
                level, self.get_valid_acc(self.epoch, level)))
            self.writer.add_scalar('valid_acc_{}'.format(level),
                                   self.get_valid_acc(level), self.epoch)
        self.reset()

    def reset(self):
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []

    def __del__(self):
        self.writer.export_scalars_to_json(
            '../logs/{}_all_scalars.json'.format(self.exp_name))
        self.writer.close()







