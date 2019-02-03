from __future__ import print_function
from models import *

from util import Dictionary, get_args
from data_loader import load_data_set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os
import numpy as np
import csv
import pdb
import pickle as pkl


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
        new_ret = torch.sum(ret)
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(data, require_grad=False):
    """Package data for training / evaluation."""
    # data = map(lambda x: json.loads(x), data)
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y] if \
        y in dictionary.word2idx.keys() else dictionary.word2idx['<unk>'], x)), data['text']))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = np.array(data['label'])
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), require_grad)
    targets = Variable(torch.from_numpy(targets), require_grad)
    return dat.t(), targets


def evaluate(att_model, data_val_or_test):
    """evaluate the model """
    att_model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    #save result into csv
    directory = "./experiments/%s/results/"%args.exp_num
    if not os.path.exists(directory):
        os.makedirs(directory)
    all_probs = []
    logSoftmax = nn.LogSoftmax()

    with open(directory+"result.csv",'w') as file:
        for batch, i in enumerate(range(0, len(data_val_or_test), args.batch_size)):
            cur_batch = data_val_or_test[i:min(len(data_val_or_test),
                                                           i + args.batch_size)]
            data, targets = package(cur_batch)
            texts = cur_batch['text']
            if args.cuda:
                data = data.cuda(args.gpu)
                targets = targets.cuda(args.gpu)
            hidden = att_model.init_hidden(data.size(1))
            output, attention = att_model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += criterion(output_flat, targets).data.cpu().numpy()
            prediction = torch.max(output_flat, 1)[1]
            probs = torch.exp(logSoftmax(output_flat))
            total_correct += torch.sum((prediction == targets).float())
            #if "%s"%data_val_or_test == "data_test":
            print("writing test results into csv file")
            pred = prediction.data.cpu().numpy()
            targ = targets.data.cpu().numpy()
            probs = probs.data.cpu().numpy()
            all_probs.append(probs)
            
            output = csv.writer(file, delimiter="\t")
            output.writerows(list(zip(pred,targ,texts)))
    pkl.dump(all_probs, open(directory+"probs.pkl",'wb'))
    return total_loss / (len(data_val_or_test) // args.batch_size), \
           total_correct.data.cpu().numpy() / len(data_val_or_test)


def train(epoch_number):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets = package(data_train[i:i + args.batch_size], False)
        if args.cuda:
            data = data.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data.cpu().numpy()

        NoneType = type(None) #check if self_attention is used
        if type(attention) is not NoneType:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += args.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data.cpu().numpy()

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {} | {}/{} batches | ms/batch {} | loss {} | pure loss {}'.format(
                epoch_number, batch, len(data_train) // args.batch_size,
                                     elapsed * 1000 / args.log_interval, total_loss / args.log_interval,
                                     total_pure_loss / args.log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()

            #            for item in model.parameters():
            #                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
            #            print model.encoder.ws2.weight.grad.data
            #            exit()
    evaluate_start_time = time.time()
    val_loss, acc = evaluate(model, data_val)
    print('-' * 89)
    fmt = '| evaluation | time: {}s | valid loss (pure) {} | Acc {}'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
    print('-' * 89)


    test_loss, test_acc = evaluate(model, data_test)
    print('-' * 89)
    fmt = '| test set result | valid loss (pure) {} | Acc {}'
    print(fmt.format(test_loss, test_acc))
    print('-' * 89)

    # Save the model, if the validation loss is the best we've seen so far.
    directory = "./experiments/%s/models/" % args.exp_num
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not best_val_loss or val_loss < best_val_loss:
        with open(directory+args.save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(directory+args.save[:-3] + '.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(directory+args.save[:-3] + \
                      '.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()


if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, "
                  "so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # # Load Data
    data_train, data_val, data_test, dictionary, c2i = \
        load_data_set(args.vocab_size, args.dataset_type, args.level)

    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,
        'nlayers': args.nlayers,
        'nhid': args.nhid,
        'ninp': args.emsize,
        'pooling': args.pooling,
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'dictionary': dictionary,
        'word-vector': args.word_vector,
        'class-number': len(c2i)
    })
    if args.cuda:
        model = model.cuda(args.gpu)

    print(args)
    I = Variable(torch.zeros(args.batch_size, args.attention_hops,
                             args.attention_hops))
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')


    # # during testing, only do here:
    if args.evaluate:
        path = '/home/ml/ksinha4/mlp/hier-class/Structured-Self-Attentive-Sentence-Embedding/experiments/0/models/models/'
        test_model = torch.load(path + args.load_model,
                                 map_location=lambda storage, loc: storage.cuda(0))
        print("finish lading test model")
        test_loss, test_acc = evaluate(test_model, data_test)
        print('-' * 89)
        print(test_loss)
        print(test_acc)
        #fmt = '| testing | valid loss (pure) {:5.4f} | Acc {:8.4f}'
        #print(fmt.format(test_loss, test_acc))
        print('-' * 89)
    else:
        try:
            for epoch in range(args.epochs):
                train(epoch)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exit from training early.')
            data_val = open(args.test_data).readlines()
            evaluate_start_time = time.time()
            test_loss, acc = evaluate()
            print('-' * 89)
            fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
            print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
            print('-' * 89)
            exit(0)
