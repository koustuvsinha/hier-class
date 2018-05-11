# Experiment on simple decoder classification

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from pprint import pprint, pformat
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
from sklearn.metrics import confusion_matrix
from hier_class.utils import data as data_utils
from hier_class.models import decoders, baselines
from hier_class.utils import constants as CONSTANTS
from hier_class.utils import model_utils as mu
from hier_class.utils.stats import Statistics
from hier_class.utils.evaluate import evaluate_test
import pdb
from hier_class.config import *

# new

@ex.automain
def train(_config, _run):
    # set seed
    torch.manual_seed(_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_config['seed'])
    # bookkeeping
    if len(_config['exp_name']) < 1:
        _config['exp_name'] = _run.start_time.strftime('%Y-%m-%d_%H:%M:%S')
    # if experiment folder exists, append timestamp after
    if os.path.exists('../../logs/' + _config['exp_name']):
        _config['exp_name'] = _config['exp_name'] + _run.start_time.strftime('%Y-%m-%d_%H:%M:%S')
    data = data_utils.Data_Utility(
        data_path=_config['data_path'],
        train_test_split=_config['train_test_split'],
        max_vocab=_config['max_vocab'],
        max_word_doc=_config['max_word_doc'],
        level = _config['level'],
        decoder_ready=_config['decoder_ready'],
        levels=_config['levels'],
        tokenization=_config['tokenization'],
        clean=_config['clean']
    )
    max_categories = _config['levels']
    if _config['level'] != -1:
        max_categories = 1
    logging.info("Loading data")
    data.load(_config['data_type'],_config['data_loc'],_config['file_name'])
    test_data = copy.deepcopy(data)
    test_data.data_mode = 'test'

    batch_size = _config['batch_size']
    gpu = _config['gpu']
    use_gpu = _config['use_gpu']
    model_params = copy.deepcopy(_config)
    cat_per_level = []
    label_size = 1
    for level in range(_config['levels']):
        nl = len(data.get_level_labels(level))
        logging.info('Classes in level {} = {}'.format(level, nl))
        cat_per_level.append(nl)
        label_size += nl
        if _config['level'] != -1:
            break
    print(cat_per_level)

    if _config['level'] != -1:
        # check if _config['level'] is not arbitrary
        if _config['level'] >= len(cat_per_level):
            raise RuntimeError("config['level'] cannot be more than config['levels']")
        logging.info("Choosing only {} level to classify".format(_config['level']))
        label_size = cat_per_level[_config['level']] + 1
    embedding = None
    if _config['use_embedding']:
        logging.info("Creating / loading word embeddings")
        embedding = data.load_embedding(_config['embedding_file'],
                                        _config['embedding_saved'],
                                        embedding_dim=_config['embedding_dim'],
                                        data_path=_config['data_path'])
    model_params.update({
        'vocab_size': len(data.word2id),
        'label_size': label_size,
        'embedding': embedding,
        'pad_token': data.word2id[CONSTANTS.PAD_WORD],
        'total_cats': sum(cat_per_level) + 1,
        'taxonomy': data.taxonomy,
        'label_sizes':cat_per_level,
        'label2id': data.label2id,
        'max_categories': max_categories,
        'gpu':_config['gpu']
    })

    logging.info("Parameters")
    logging.info(pformat(_config))

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
    if _config['baseline']:
        assert _config['level'] != -1
        label_weights = data.calculate_weights(_config['level'])
        #logging.info("Label weights")
        #logging.info(label_weights)
        label_weights = torch.FloatTensor(label_weights)
        label_weights = label_weights.cuda(gpu)
        if _config['baseline'] == 'fast':
            model = baselines.FastText(**model_params)
        elif _config['baseline'] == 'bilstm':
            model = baselines.BiLSTM_MLP(**model_params)
        else:
            raise NotImplementedError("Baseline not implemented")
    else:
        label_weights = [1.0]
        for i in range(_config['levels']):
            label_weights.extend(data.calculate_weights(i+1))
        #logging.info("Label weights")
        #logging.info(label_weights)
        label_weights = torch.FloatTensor(label_weights)
        label_weights = label_weights.cuda(gpu)
        model = decoders.SimpleMLPDecoder(**model_params)
    print(model)
    m_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _config['optimizer']
    num_params = sum([np.prod(p.size()) for p in m_params])
    logging.info("Model parameters : {}".format(num_params))
    if optimizer == 'adam':
        optimizer = optim.Adam(m_params, lr=_config['lr'], weight_decay=_config['weight_decay'])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(m_params, lr=_config['lr'], weight_decay=_config['weight_decay'])
    elif optimizer == 'sgd':
        optimizer = optim.SGD(m_params, lr=_config['lr'], momentum=_config['momentum'],
                              weight_decay=_config['weight_decay'])
    else:
        raise NotImplementedError()
    if use_gpu:
        model = model.cuda(gpu)

    # set learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='max',
                                     factor=_config['lr_factor'],
                                     threshold=_config['lr_threshold'],
                                     patience=_config['lr_patience'], verbose=True)

    tf_ratio = _config['tf_ratio']
    logging.info("Starting to train")
    pytorch_version = torch.__version__
    logging.info("Using pytorch version : {}".format(pytorch_version))
    epochs = _config['epochs']
    max_levels = _config['levels']
    if _config['level'] != -1:
        max_levels = 1
    stats = Statistics(batch_size,max_levels,_config['exp_name'],data=data,n_heads=_config['n_heads'],level=_config['level'])
    logging.info("With focus : {}".format(_config['loss_focus']))
    all_step = 0
    for epoch in range(epochs):
        print(epoch)
        stats.next()
        logging.info("Getting data")
        logging.info("Num Train Rows: {}".format(len(data.train_indices)))
        logging.info("Num Test Rows: {}".format(len(data.test_indices)))
        logging.info("TF Ratio: {}".format(tf_ratio))
        train_data_loader = torch.utils.data.DataLoader(dataset=data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn,
                                                  num_workers=4)
        train_data_iter = iter(train_data_loader)

        test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=data_utils.collate_fn,
                                                  num_workers=4)
        test_data_iter = iter(test_data_loader)
        logging.info("Got data")
        model.train()
        loss = None
        for src_data, src_lengths, src_labels, src_index in train_data_iter:
            labels = Variable(torch.LongTensor(src_labels))
            #cat_labels = Variable(torch.LongTensor(cat_labels))
            src_data = Variable(src_data)
            if use_gpu:
                src_data = src_data.cuda(gpu)
                labels = labels.cuda(gpu)
            #    cat_labels = cat_labels.cuda(gpu)
            optimizer.zero_grad()
            loss, accs, *_ = model.batchNLLLoss(src_data, src_lengths, labels,
                                            tf_ratio=tf_ratio,
                                            loss_focus=_config['loss_focus'],
                                            loss_weights=label_weights,
                                            max_categories=max_categories,
                                            target_level=1,
                                            attn_penalty_coeff=_config['attn_penalty_coeff'],
                                            renormalize=_config['renormalize'])
            loss.backward()
            #m_params = [p for p in model.parameters() if p.requires_grad]
            #nn.utils.clip_grad_norm(m_params, _config['clip_grad'])
            optimizer.step()
            stats.update_train(loss.data[0], accs)
            ## free up memory
            del labels
            del src_data
            del loss
            del accs
            if _config['debug']:
                break
        ## validate
        model.eval()
        ## store the attention weights and words in a separate file for
        ## later visualization
        storage = []
        for src_data, src_lengths, src_labels, src_index in test_data_iter:
            labels =  Variable(torch.LongTensor(src_labels), volatile=True)
            #cat_labels = Variable(torch.LongTensor(cat_labels))
            src_data = Variable(src_data, volatile=True)
            if use_gpu:
                src_data = src_data.cuda(gpu)
            #    cat_labels = cat_labels.cuda(gpu)
            labels = labels.cuda(gpu)
            loss, accs, attns, preds, correct, correct_confs, incorrect_confs = model.batchNLLLoss(src_data, src_lengths, labels,
                                            tf_ratio=_config['validation_tf'],
                                            loss_focus=_config['loss_focus'],
                                            loss_weights=label_weights,
                                            max_categories=max_categories,
                                            target_level=1,
                                            attn_penalty_coeff=_config['attn_penalty_coeff'],
                                            renormalize=_config['renormalize'])
            stats.update_validation(loss.data[0],accs, attn=attns, src=src_data, preds=preds, correct=correct,
                                    correct_confs=correct_confs, incorrect_confs=incorrect_confs)
        stats.log_loss()
        lr_scheduler.step(stats.get_valid_acc(0))
        ## anneal
        tf_ratio = tf_ratio * _config['tf_anneal']
        ## saving model
        mu.save_model(model,0,0,_config['exp_name'],model_params)
    ## Evaluate Testing data
    model.eval()
    evaluate_test(model, data, _config['test_file_name'],_config['test_output_name'],_config)





