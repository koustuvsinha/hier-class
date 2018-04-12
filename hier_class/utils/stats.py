# class to maintain statistics and logging utils

import torch
import numpy as np
import re
import itertools
from textwrap import wrap
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        self.correct_labels = []
        self.predicted_labels = []
        self.correct_confs = []
        self.incorrect_confs = []
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

    def update_validation(self, validation_loss, validation_accuracy, attn=None, src=None,
                          preds=None, correct=None, correct_confs=None,incorrect_confs=None, **kwargs):
        self.validation_loss.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)
        self.predicted_labels.append(preds)
        self.correct_labels.append(correct)
        self.correct_confs.append(correct_confs)
        self.incorrect_confs.append(incorrect_confs)

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

    def get_valid_conf(self, level=0):
        return np.mean([tr[level] for tr in self.correct_confs]), np.mean([tr[level] for tr in self.incorrect_confs])

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
                level, self.get_valid_acc(level)))
            self.writer.add_scalar('valid_acc_{}'.format(level),
                                   self.get_valid_acc(level), self.epoch)
            valid_conf_true, valid_conf_false = self.get_valid_conf(level)
            logging.info("Validation correct confidence for level {} : {}".format(
                level, valid_conf_true
            ))
            logging.info("Validation incorrect confidence for level {} : {}".format(
                level, valid_conf_false
            ))
            self.writer.add_scalar('valid_conf_{}'.format(level), valid_conf_true, self.epoch)
            self.writer.add_scalar('valid_conf_{}'.format(level), valid_conf_false, self.epoch)
            if level==0:
                logging.info("Saving confusion matrix for level {}".format(level))
                correct_labels = flatten([tr[level] for tr in self.correct_labels])
                predicted_labels = flatten([tr[level] for tr in self.predicted_labels])
                conf_matrix = plot_confusion_matrix(correct_labels, predicted_labels, range(0, max(correct_labels)))
                self.writer.add_image('Confusion_Matrix_Level_{}'.format(level),conf_matrix,self.epoch)
        self.reset()

    def reset(self):
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.correct_confs = []
        self.incorrect_confs = []

    def __del__(self):
        self.writer.export_scalars_to_json(
            '../../logs/{}_all_scalars.json'.format(self.exp_name))
        self.writer.close()


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', normalize=False):
    '''
    Source: https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    #classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    #classes = ['\n'.join(wrap(l, 40)) for l in classes]
    classes = [str(lb) for lb in labels]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def flatten(x):
    '''
    Flatten a list of list
    :param x:
    :return:
    '''
    return [b for a in x for b in a]





