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
import os
import logging
import pdb


from codes.utils import model_utils as mu


class Statistics():
    """
    Class to collect training and validation statistics.
    Also collect validation samples and attention for later inspection
    """
    def __init__(self, batch_size=0, max_levels=3, exp_name='', data=None, n_heads=[], level=-1):
        self.epoch = -1
        self.step = 0
        self.train_accuracy = []
        self.batch_size = batch_size
        self.max_levels = max_levels
        self.exp_name = exp_name
        self.data = data
        self.n_heads = n_heads
        self.level = level
        self.base_dir = str(os.path.dirname(os.path.realpath(__file__)).split('codes')[0])
        self.log_dir = os.path.join(self.base_dir, 'logs')
        writer_dir = os.path.join(self.log_dir,exp_name)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        self.writer = SummaryWriter(log_dir=writer_dir)
        self.output = {} # json file to store validation examples. should contain true and predicted labels (actual class names), validation examples, and generated attentions per epoch.
        self.output['train_indices'] = data.train_indices
        self.output['val_indices'] = data.test_indices
        self.reset()

    def next(self):
        """Update the epoch and start timer"""
        self.epoch += 1
        self.step = 0
        self.calc_start = time.time()
        self.output[self.epoch] = {'attentions':[], 'val_indices':[], 'predictions':[]}
        save_path_base = mu.create_save_dir(self.exp_name)
        #json.dump(self.output, open(save_path_base + '/val_logs.txt','w'))
        logging.info("Epoch : {}".format(self.epoch))


    def update_train(self, train_loss, train_accuracy, log_loss=0):
        self.train['train_loss'].append(train_loss)
        self.train['train_accuracy'].append(train_accuracy)
        self.train['train_log_loss'].append(log_loss)
        self.step +=1

    def update_validation(self, validation_loss, validation_accuracy, attn=None, src=None,
                          preds=None, correct=None, correct_confs=None,incorrect_confs=None,
                          log_loss=0, mode='exact', **kwargs):
        self.val[mode]['validation_loss'].append(validation_loss)
        self.val[mode]['validation_log_loss'].append(log_loss)
        self.val[mode]['validation_accuracy'].append(validation_accuracy)
        self.val[mode]['predicted_labels'].append(preds)
        self.val[mode]['correct_labels'].append(correct)
        self.val[mode]['correct_confs'].append(correct_confs)
        self.val[mode]['incorrect_confs'].append(incorrect_confs)

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


    def get_train_acc(self, level=0):
        train_acc = self.train['train_accuracy']
        return np.mean([tr[level] for tr in train_acc])

    def get_valid_acc(self, level=0, mode='exact'):
        valid_acc = self.val[mode]['validation_accuracy']
        return np.mean([tr[level] for tr in valid_acc])

    def get_valid_conf(self, level=0, mode='exact'):
        valid_conf_false = np.mean([tr[level] for tr in self.val[mode]['incorrect_confs']])
        valid_conf_true = np.mean([tr[level] for tr in self.val[mode]['correct_confs']])
        return valid_conf_true, valid_conf_false

    def log_loss(self):
        time_taken = time.time() - self.calc_start
        m, s = divmod(time_taken, 60)
        h, m = divmod(m, 60)
        logging.info('Time taken: {}:{}:{}'.format(h,m,s))
        logging.info("After Epoch {}".format(self.epoch))
        train_loss = np.mean(self.train['train_loss'])
        logging.info("Train Loss : {}".format(train_loss))
        logging.info("Train Log Loss : {}".format(np.mean(self.train['train_log_loss'])))
        self.writer.add_scalar('train_loss',train_loss,self.epoch)
        valid_loss_exact = np.mean(self.val['exact']['validation_loss'])
        valid_loss_overall = np.mean(self.val['overall']['validation_loss'])
        logging.info("Validation Loss : Exact : {}, Overall : {}".format(valid_loss_exact, valid_loss_overall))
        self.writer.add_scalar('validation_loss_exact', valid_loss_exact, self.epoch)
        self.writer.add_scalar('validation_loss_overall', valid_loss_overall, self.epoch)
        for level in range(self.max_levels):
            logging.info("Train accuracy for level {} : {}".format(
                level, self.get_train_acc(level)))
            self.writer.add_scalar('train_acc_{}'.format(level),
                                   self.get_train_acc(level), self.epoch)
            logging.info("Validation accuracy, Mode: exact, for level {} : {}".format(
                level, self.get_valid_acc(level, mode='exact')))
            self.writer.add_scalar('valid_acc_exact_{}'.format(level),
                                   self.get_valid_acc(level, mode='exact'), self.epoch)
            valid_conf_true_e, valid_conf_false_e = self.get_valid_conf(level, mode='exact')
            valid_conf_true_o, valid_conf_false_o = self.get_valid_conf(level, mode='overall')
            logging.info("Validation accuracy, Mode: overall, for level {} : {}".format(
                level, self.get_valid_acc(level, mode='overall')))
            self.writer.add_scalar('valid_acc_exact_{}'.format(level),
                                   self.get_valid_acc(level, mode='overall'), self.epoch)

            logging.info("Validation correct confidence for level {} :  Exact : {}, Overall : {}".format(
                level, valid_conf_true_e, valid_conf_true_o
            ))
            logging.info("Validation incorrect confidence for level {} : Exact : {}, Overall : {}".format(
                level, valid_conf_false_e, valid_conf_false_o
            ))
            self.writer.add_scalar('valid_conf_exact_{}'.format(level), valid_conf_true_e, self.epoch)
            self.writer.add_scalar('valid_conf_exact_{}'.format(level), valid_conf_false_e, self.epoch)
            self.writer.add_scalar('valid_conf_overall_{}'.format(level), valid_conf_true_o, self.epoch)
            self.writer.add_scalar('valid_conf_overall_{}'.format(level), valid_conf_false_o, self.epoch)

            """
            if level==0:
                logging.info("Saving confusion matrix for level {}".format(level))
                correct_labels = flatten([tr[level] for tr in self.correct_labels])
                predicted_labels = flatten([tr[level] for tr in self.predicted_labels])
                logging.info("Min predicted label {}".format(min(predicted_labels)))
                logging.info("Max predicted label {}".format(max(predicted_labels)))
                conf_matrix = plot_confusion_matrix(correct_labels, predicted_labels, range(1, max(correct_labels) + 1))
                self.writer.add_image('Confusion_Matrix_Level_{}'.format(level),conf_matrix,self.epoch)
            """
        #self.reset()

    def reset(self):
        self.train = {
            'train_loss': [],
            'train_accuracy': [],
            'train_log_loss': []
        }
        self.val = {
            'exact': {
                'validation_loss': [],
                'validation_log_loss': [],
                'validation_accuracy': [],
                'predicted_labels': [],
                'correct_labels': [],
                'correct_confs': [],
                'incorrect_confs': []
            },
            'overall': {
                'validation_loss': [],
                'validation_log_loss': [],
                'validation_accuracy': [],
                'predicted_labels': [],
                'correct_labels': [],
                'correct_confs': [],
                'incorrect_confs': []
            }
        }

    def __del__(self):
        log_path = os.path.join(self.log_dir, '{}_all_scalars.json'.format(self.exp_name))
        self.writer.export_scalars_to_json(log_path)
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





