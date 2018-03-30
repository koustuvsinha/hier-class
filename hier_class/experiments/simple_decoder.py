# Experiment on simple decoder classification

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
from hier_class.utils import model_utils as mu
from hier_class.utils.stats import Statistics
import pdb

ex = Experiment()

@ex.config
def exp_config():
    gpu = 0
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
    data_loc = '/home/ml/ksinha4/datasets/data_WIKI'
    file_name = 'full_docs_2.csv'
    #data_loc = '/home/ml/ksinha4/datasets/data_WOS/WebOfScience/WOS46985'
    tokenization = 'word'
    batch_size = 16
    epochs = 60
    level = -1
    levels = 3
    cat_emb_dim = 64
    tf_ratio=0.5
    tf_anneal=0.8
    validation_tf = 0
    weight_decay=1e-6
    temperature = 1
    loss_focus = [1,1,1]
    label_weights = [1,1,1]
    dynamic_dictionary = True
    max_vocab = 100000
    max_word_doc = -1
    decoder_ready = True

@ex.automain
def train(_config, _run):
    # bookkeeping
    if len(_config['exp_name']) < 1:
        _config['exp_name'] = _run.start_time.strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter(log_dir='../../logs/' + _config['exp_name'])
    data = data_utils.Data_Utility(
        exp_name=_config['exp_name'],
        train_test_split=_config['train_test_split'],
        max_vocab=_config['max_vocab'],
        max_word_doc=_config['max_word_doc'],
        level = _config['level'],
        decoder_ready=_config['decoder_ready']
    )
    max_categories = _config['levels']
    if _config['level'] != -1:
        max_categories = 1
    logging.info("Loading data")
    data.load(_config['data_type'],_config['data_loc'],_config['file_name'],_config['tokenization'])
    test_data = copy.deepcopy(data)
    test_data.data_mode = 'test'

    batch_size = _config['batch_size']
    gpu = _config['gpu']
    use_gpu = _config['use_gpu']
    model_params = copy.deepcopy(_config)
    cat_per_level = []
    for level in range(_config['levels']):
        nl = len(data.get_level_labels(level))
        logging.info('Classes in level {} = {}'.format(level, nl))
        cat_per_level.append(nl)
    label_size = data.decoder_num_labels
    if _config['level'] != -1:
        # check if _config['level'] is not arbitrary
        if _config['level'] >= len(cat_per_level):
            raise RuntimeError("config['level'] cannot be more than config['levels']")
        logging.info("Choosing only {} level to classify".format(_config['level']))
        label_size = cat_per_level[_config['level']]
    model_params.update({
        'vocab_size': len(data.word2id),
        'label_size': label_size,
        'pad_token': data.word2id[CONSTANTS.PAD_WORD],
        'total_cats': sum(cat_per_level) + 1,
        'taxonomy': data.taxonomy,
        'label_sizes':cat_per_level,
        'label2id': data.label2id,
        'gpu':_config['gpu']
    })

    ## calculate label weights
    ## for level1 labels = 1.0
    ## for level2 labels = 0.8
    ## for level3 labels = 0.6
    level2w = {}
    for i,lb in enumerate(_config['label_weights']):
        level2w[i] = lb
    label_weights = []
    if _config['level'] == -1:
        label_weights = [0.0]
    for level in range(_config['levels']):
        if _config['level'] != -1 and _config['level'] != level:
            continue
        labels = list(sorted(data.get_level_labels(level)))
        for lb in labels:
            label_weights.append(level2w[level])
    label_weights = torch.FloatTensor(label_weights)
    label_weights = label_weights.cuda(gpu)


    #model = decoders.SimpleDecoder(**model_params)
    model = decoders.SimpleMLPDecoder(**model_params)
    print(model)
    m_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _config['optimizer']
    if optimizer == 'adam':
        optimizer = optim.Adam(m_params, lr=_config['lr'], weight_decay=_config['weight_decay'])
    else:
        raise NotImplementedError()
    if use_gpu:
        model = model.cuda(gpu)

    tf_ratio = _config['tf_ratio']
    logging.info("Starting to train")
    pytorch_version = torch.__version__
    logging.info("Using pytorch version : {}".format(pytorch_version))
    epochs = _config['epochs']
    max_levels = _config['levels']
    if _config['level'] != -1:
        max_levels = 1
    stats = Statistics(batch_size,max_levels,_config['exp_name'])
    logging.info("With focus : {}".format(_config['loss_focus']))
    all_step = 0
    for epoch in range(epochs):
        stats.next()
        logging.info("Getting data")
        logging.info("Num Train Rows: {}".format(len(data.train_indices)))
        logging.info("Num Test Rows: {}".format(len(data.test_indices)))
        logging.info("TF Ratio: {}".format(tf_ratio))
        train_data_loader = torch.utils.data.DataLoader(dataset=data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn,
                                                  num_workers=8)
        train_data_iter = iter(train_data_loader)

        test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn,
                                                  num_workers=8)
        test_data_iter = iter(test_data_loader)
        logging.info("Got data")
        model.train()
        for src_data, src_lengths, src_labels in train_data_iter:
            labels = Variable(torch.LongTensor(src_labels))
            #cat_labels = Variable(torch.LongTensor(cat_labels))
            src_data = Variable(src_data)
            if use_gpu:
                src_data = src_data.cuda(gpu)
                labels = labels.cuda(gpu)
            #    cat_labels = cat_labels.cuda(gpu)
            optimizer.zero_grad()
            loss, accs, _ = model.batchNLLLoss(src_data, src_lengths, labels,
                                            tf_ratio=tf_ratio,
                                            loss_focus=_config['loss_focus'],
                                            loss_weights=label_weights,
                                            max_categories=max_categories)
            loss.backward()
            #m_params = [p for p in model.parameters() if p.requires_grad]
            #nn.utils.clip_grad_norm(m_params, 5)
            optimizer.step()
            stats.update_train(loss.data[0], accs)
            #break
        ## validate
        model.eval()
        ## store the attention weights and words in a separate file for
        ## later visualization
        storage = []
        for src_data, src_lengths, src_labels in test_data_iter:
            labels =  Variable(torch.LongTensor(src_labels), volatile=True)
            #cat_labels = Variable(torch.LongTensor(cat_labels))
            src_data = Variable(src_data, volatile=True)
            if use_gpu:
                src_data = src_data.cuda(gpu)
            #    cat_labels = cat_labels.cuda(gpu)
            labels = labels.cuda(gpu)
            loss, accs, attns = model.batchNLLLoss(src_data, src_lengths, labels,
                                            tf_ratio=_config['validation_tf'],
                                            loss_focus=_config['loss_focus'],
                                            loss_weights=label_weights,
                                            max_categories=max_categories)

            #src_d = src_data.data
            stats.update_validation(loss.data[0],accs)
        stats.log_loss()
        ## anneal
        tf_ratio = tf_ratio * _config['tf_anneal']
        ## saving model
        mu.save_model(model,0,0,_config['exp_name'],model_params)

    stats.cleanup()




