# Experiment on simple decoder classification

# CUDA_VISIBLE_DEVICES=4 python simple_decoder.py with exp_name=decode_context batch_size=64
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import logging
from sacred import Experiment
import logging
from tensorboardX import SummaryWriter
import time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import sys
import os
import copy
import json
from tqdm import tqdm
from hier_class.utils import data as data_utils
from hier_class.models import decoders
from hier_class.utils import constants as CONSTANTS

ex = Experiment()
writer = None

@ex.config
def exp_config():
    gpu = 2
    use_gpu = True
    exp_name = ''
    embedding_dim = 300
    use_embedding = False
    fix_embeddings = False
    embedding_file = '/home/ml/ksinha4/word_vectors/glove/glove.840B.300d.txt'
    embedding_saved = 'wos_embeddings.mod'
    load_model = False
    load_model_path = ''
    save_name = 'model_epoch_{}_step_{}.mod'
    optimizer = 'adam'
    lr = 1e-3
    log_interval = 200
    save_interval = 1000
    train_test_split = 0.8
    data_type = 'WIKI'
    data_loc = '/home/ml/ksinha4/mlp/hier-class/data/'
    #data_loc = '/home/ml/ksinha4/datasets/data_WOS/WebOfScience/WOS46985'
    tokenization = 'word'
    batch_size = 16
    epochs = 40
    level = 2
    cat_emb_dim = 64

@ex.automain
def train(_config, _run):
    # bookkeeping
    # if len(_config['exp_name']) < 1:
    #     _config['exp_name'] = _run.start_time.strftime('%Y-%m-%d_%H:%M:%S')
    #writer = SummaryWriter(log_dir='../../logs/' + _config['exp_name'])
    data = data_utils.Data_Utility(
        #exp_name=_config['exp_name'],
        train_test_split=_config['train_test_split'],
        decoder_ready=True
    )
    logging.info("Loading data")
    data.load(_config['data_type'],_config['data_loc'],_config['tokenization'])
    batch_size = _config['batch_size']
    gpu = _config['gpu']
    use_gpu = _config['use_gpu']
    model_params = copy.deepcopy(_config)
    logging.info('Classes in level {} = {}'.format(_config['level'], len(data.get_level_labels(_config['level']))))
    model_params.update({
        'vocab_size': len(data.word2id),
        'label_size': data.decoder_num_labels,
        'pad_token': data.word2id[CONSTANTS.PAD_WORD]
    })

    model = decoders.SimpleDecoder(**model_params)
    print(model)
    m_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _config['optimizer']
    if optimizer == 'adam':
        optimizer = optim.Adam(m_params, lr=_config['lr'])
    else:
        raise NotImplementedError()
    if use_gpu:
        model = model.cuda(gpu)


    logging.info("Starting to train")
    pytorch_version = torch.__version__
    logging.info("Using pytorch version : {}".format(pytorch_version))
    epochs = _config['epochs']
    all_step = 0
    calc_start = time.time()
    for epoch in range(epochs):
        train_loss = []
        validation_loss = []
        train_acc = []
        validation_acc = []
        logging.info("Getting data")
        logging.info("Num Train Rows: {}".format(len(data.train_indices)))
        logging.info("Num Test Rows: {}".format(len(data.test_indices)))
        train_data_loader = torch.utils.data.DataLoader(dataset=data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn)
        train_data_iter = iter(train_data_loader)
        test_data = copy.deepcopy(data)
        test_data.data_mode = 'test'
        test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn)
        test_data_iter = iter(test_data_loader)
        for src_data, src_lengths, src_labels in train_data_iter:
            labels = Variable(torch.LongTensor(src_labels))
            src_data = Variable(src_data)
            if use_gpu:
                src_data = src_data.cuda(gpu)
                labels = labels.cuda(gpu)
            optimizer.zero_grad()
            loss, acc = model.batchNLLLoss(src_data, src_lengths, labels,tf_ratio=0.5)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data[0])
            train_acc.append(acc)
        ## validate
        for src_data, src_lengths, src_labels in test_data_iter:
            labels =  Variable(torch.LongTensor(src_labels))
            src_data = Variable(src_data)
            if use_gpu:
                src_data = src_data.cuda(gpu)
            labels = labels.cuda(gpu)
            loss, acc = model.batchNLLLoss(src_data, src_lengths, labels, tf_ratio=0)
            validation_loss.append(loss.data[0])
            validation_acc.append(acc)
        print('After Epoch {}'.format(epoch))
        print('Train Loss {}'.format(np.mean(train_loss)))
        print('Validation loss {}'.format(np.mean(validation_loss)))
        print('Train accuracy {}'.format(np.mean(train_acc)))
        print('Validation accuracy {}'.format(np.mean(validation_acc)))



