# class to maintain statistics and logging utils

import numpy as np
from tensorboardX import SummaryWriter
import time
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Statistics():
    def __init__(self, batch_size=0, max_levels=3, exp_name=''):
        self.epoch = -1
        self.step = 0
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.batch_size = batch_size
        self.max_levels = max_levels
        self.exp_name = exp_name
        self.writer = SummaryWriter(log_dir='../../logs/' + exp_name)

    def next(self):
        """Update the epoch and start timer"""
        self.epoch += 1
        self.step = 0
        self.calc_start = time.time()
        logging.info("Epoch : {}".format(self.epoch))

    def update_train(self, train_loss, train_accuracy):
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.step +=1

    def update_validation(self, validation_loss, validation_accuracy):
        self.validation_loss.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)

    def get_train_loss(self, epoch=0):
        start_index = epoch*self.batch_size
        return np.mean(self.train_loss[start_index:])

    def get_valid_loss(self, epoch=0):
        start_index = epoch * self.batch_size
        return np.mean(self.validation_loss[start_index:])

    def get_train_acc(self, epoch=0, level=0):
        start_index = epoch * self.batch_size
        train_acc = self.train_accuracy[start_index:]
        return np.mean([tr[level] for tr in train_acc])

    def get_valid_acc(self, epoch=0, level=0):
        start_index = epoch * self.batch_size
        valid_acc = self.validation_accuracy[start_index:]
        return np.mean([tr[level] for tr in valid_acc])

    def log_loss(self):
        time_taken = time.time() - self.calc_start
        logging.info('Time taken: {}'.format(time_taken * 1000))
        logging.info("After Epoch {}".format(self.epoch))
        logging.info("Train Loss : {}".format(self.get_train_loss(self.epoch)))
        self.writer.add_scalar('train_loss',self.get_train_loss(self.epoch),self.epoch)
        logging.info("Validation Loss : {}".format(self.get_valid_loss(self.epoch)))
        self.writer.add_scalar('validation_loss', self.get_valid_loss(self.epoch), self.epoch)
        for level in range(self.max_levels):
            logging.info("Train accuracy for level {} : {}".format(
                level, self.get_train_acc(self.epoch, level)))
            self.writer.add_scalar('train_acc_{}'.format(level),
                                   self.get_train_acc(self.epoch, level), self.epoch)
            logging.info("Validation accuracy for level {} : {}".format(
                level, self.get_valid_acc(self.epoch, level)))
            self.writer.add_scalar('valid_acc_{}'.format(level),
                                   self.get_valid_acc(self.epoch, level), self.epoch)

    def cleanup(self):
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.writer.export_scalars_to_json(
            '../logs/{}_all_scalars.json'.format(__file__))
        self.writer.close()







