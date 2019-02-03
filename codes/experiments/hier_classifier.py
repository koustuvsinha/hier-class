# Experiment on simple decoder classification

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import numpy as np
from pprint import pprint, pformat
import os
import copy
from codes.utils import data as data_utils
from codes.models import decoders, baselines
from codes.utils import constants as CONSTANTS
from codes.utils import model_utils as mu
from codes.utils.stats import Statistics
from codes.utils.evaluate import evaluate_test


# select device automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_experiment(config, _run):
    # set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    # bookkeeping
    if len(config['exp_name']) < 1:
        config['exp_name'] = _run.start_time.strftime('%Y_%m_%d_%H_%M_%S')
    # if experiment folder exists, append timestamp after
    base_dir = str(os.path.dirname(os.path.realpath(__file__)).split('codes')[0])
    exp_log_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(os.path.join(exp_log_dir, config['exp_name'])):
        config['exp_name'] = os.path.join(config['exp_name'], _run.start_time.strftime('%Y_%m_%d_%H_%M_%S'))
    data = data_utils.Data_Utility(config)
    max_categories = config['levels']
    if config['level'] != -1:
        max_categories = 1
    logging.info("Loading data")
    data.load()

    batch_size = config['batch_size']
    gpu = config['gpu']
    use_gpu = config['use_gpu']
    model_params = copy.deepcopy(config)
    cat_per_level = []
    label_size = 1
    for level in range(config['levels']):
        nl = len(data.get_level_labels(level))
        logging.info('Classes in level {} = {}'.format(level, nl))
        cat_per_level.append(nl)
        label_size += nl
        if config['level'] != -1:
            break
    print(cat_per_level)

    if config['level'] != -1:
        # check if _config['level'] is not arbitrary
        if config['level'] >= len(cat_per_level):
            raise RuntimeError("config['level'] cannot be more than config['levels']")
        logging.info("Choosing only {} level to classify".format(config['level']))
        label_size = cat_per_level[config['level']] + 1
    embedding = None
    if config['use_embedding']:
        logging.info("Creating / loading word embeddings")
        embedding = data.load_embedding(config['embedding_file'],
                                        config['embedding_saved'],
                                        embedding_dim=config['embedding_dim'],
                                        data_path=config['data_path'])
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
        'gpu':config['gpu']
    })

    logging.info("Parameters")
    logging.info(pformat(config))

    ## calculate label weights
    ## for level1 labels = 1.0
    ## for level2 labels = 0.8
    ## for level3 labels = 0.6
    level2w = {}
    for i,lb in enumerate(config['label_weights']):
        level2w[i] = lb
    label_weights = []
    if config['level'] == -1:
        label_weights = [0.0]
    for level in range(config['levels']):
        if config['level'] != -1 and config['level'] != level:
            continue
        labels = list(sorted(data.get_level_labels(level)))
        for lb in labels:
            label_weights.append(level2w[level])
    #label_weights = torch.FloatTensor(label_weights)
    #label_weights = label_weights.cuda(gpu)


    #model = decoders.SimpleDecoder(**model_params)
    if config['baseline']:
        assert config['level'] != -1
        label_weights = data.calculate_weights(config['level'])
        #logging.info("Label weights")
        #logging.info(label_weights)
        label_weights = torch.FloatTensor(label_weights).to(device)
        if config['baseline'] == 'fast':
            model = baselines.FastText(**model_params)
        elif config['baseline'] == 'bilstm':
            model = baselines.BiLSTM_MLP(**model_params)
        else:
            raise NotImplementedError("Baseline not implemented")
    else:
        label_weights = [1.0]
        for i in range(config['levels']):
            label_weights.extend(data.calculate_weights(i+1))
        #logging.info("Label weights")
        #logging.info(label_weights)
        label_weights = torch.FloatTensor(label_weights).to(device)
        if config['model_type'] == 'attentive':
            model = decoders.AttentiveHierarchicalClassifier(**model_params)
        elif config['model_type'] == 'pooling':
            model = decoders.PooledHierarchicalClassifier(**model_params)

    print(model)

    m_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = config['optimizer']
    num_params = sum([np.prod(p.size()) for p in m_params])
    logging.info("Model parameters : {}".format(num_params))
    if optimizer == 'adam':
        optimizer = optim.Adam(m_params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(m_params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif optimizer == 'sgd':
        optimizer = optim.SGD(m_params, lr=config['lr'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError()

    model = model.to(device)

    # set learning rate scheduler
    if config['lr_scheduler'] == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode='min',
                                         factor=config['lr_factor'],
                                         threshold=config['lr_threshold'],
                                         patience=config['lr_patience'],
                                         cooldown=1,
                                         verbose=True)
    elif config['lr_scheduler'] == 'sltr':
        lr_scheduler = mu.SLTR(epochs=config['epochs'],
                           batch_size=config['batch_size'],
                           num_train=len(data.train_indices))
    else:
        raise NotImplementedError("lr_scheduler {} not implemented".format(config['lr_scheduler']))

    # create trainer
    trainer = decoders.Trainer(model=model, loss_weights=label_weights,
                               **model_params)

    tf_ratio = config['tf_ratio']
    logging.info("Starting to train")
    pytorch_version = torch.__version__
    logging.info("Using pytorch version : {}".format(pytorch_version))
    epochs = config['epochs']
    max_levels = config['levels']
    if config['level'] != -1:
        max_levels = 1
    stats = Statistics(batch_size, max_levels, config['exp_name'], data=data, n_heads=config['n_heads'], level=config['level'])
    logging.info("With focus : {}".format(config['loss_focus']))
    all_step = 0
    for epoch in range(epochs):
        stats.next()
        logging.info("Getting data")
        logging.info("Num Train Rows: {}".format(len(data.train_indices)))
        logging.info("Num Test Rows: {}".format(len(data.test_indices)))
        logging.info("TF Ratio: {}".format(tf_ratio))
        train_data_loader = data.get_dataloader(mode='train')
        model.train()
        loss = None
        for batch_idx, batch in enumerate(train_data_loader):
            if config['lr_scheduler'] == 'sltr':
                optimizer = lr_scheduler.step(optimizer)
            optimizer.zero_grad()
            batch.to_device(device)
            (loss, log_loss), accs, attns, *_ = trainer.batchNLLLoss(batch.inp, batch.inp_lengths,
                                            batch.outp,mode='train', tf_ratio=config['tf_ratio'])
            torch.cuda.empty_cache()
            loss.backward()
            m_params = [p for p in model.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm(m_params, config['clip_grad'])
            optimizer.step()
            stats.update_train(loss.item(), accs, log_loss=log_loss.item())
            ## free up memory
            del batch
            del loss
            del accs
            del attns
            if config['debug']:
                break
        ## validate
        model.eval()
        ## store the attention weights and words in a separate file for
        ## later visualization
        storage = []
        test_data_loader = data.get_dataloader(mode='test')
        valid_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                batch.to_device(device)
                ## overall - teacher_forcing false
                (loss, log_loss), accs, attns, preds, correct, correct_confs, incorrect_confs,_ = trainer.batchNLLLoss(
                    batch.inp, batch.inp_lengths, batch.outp, mode='infer',overall=True)
                stats.update_validation(loss.item(),accs, attn=attns, src=batch.inp, preds=preds, correct=correct,
                                        correct_confs=correct_confs,
                                        incorrect_confs=incorrect_confs,
                                        log_loss=log_loss.item(),
                                        mode='overall')
                valid_losses.append(loss.item())

                ## exact - teacher_forcing true
                (loss, log_loss), accs, attns, preds, correct, correct_confs, incorrect_confs,_ = trainer.batchNLLLoss(
                    batch.inp, batch.inp_lengths, batch.outp, mode='infer', overall=False)
                stats.update_validation(loss.item(), accs, attn=attns, src=batch.inp, preds=preds, correct=correct,
                                        correct_confs=correct_confs,
                                        incorrect_confs=incorrect_confs,
                                        log_loss=log_loss.item(),
                                        mode='exact')

                valid_losses.append(loss.item())

                del batch
                del loss
                del accs
                del attns
                del preds
                if config['debug']:
                    break
            stats.log_loss()
            valid_loss = np.mean(valid_losses)
            #valid_acc_lr = stats.get_valid_acc(config['levels'] - 1)
            #print('valid_acc_lr {}'.format(valid_acc_lr))
            if config['lr_scheduler'] == 'plateau':
                lr_scheduler.step(valid_loss)
            stats.reset()
            ## anneal
            tf_ratio = tf_ratio * config['tf_anneal']
            ## saving model
            mu.save_model(model, epoch, 0, config['exp_name'], model_params)
    ## Evaluate Testing data
    ## trainer.model.eval()
    ## evaluate_test(trainer, data, config['test_file_name'], config['test_output_name'], config)





