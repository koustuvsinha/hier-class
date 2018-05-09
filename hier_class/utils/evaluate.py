# Evaluate a saved model with respect to testing data
import torch
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
import argparse
from hier_class.models import decoders
from hier_class.utils import data as data_utils
from hier_class.utils import constants
import pdb
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_args():

    ## arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--exp", type=str, help="experiment to load", default="wiki_normalize_experiment2018-04-21_18:43:17")
    parser.add_argument("-m","--model", type=str, help="model to load", default="model_epoch_0_step_0.mod")
    parser.add_argument("-f","--file", type=str, help="testing file to load", default="full_docs_2_test.csv")
    parser.add_argument("-o","--output", type=str, help="file to write the output", default="output.csv")
    parser.add_argument("-c","--confidence", type=float, help="confidence to measure pruned accuracy", default=0.0)
    parser.add_argument("-n","--num", type=int, help="number of evals (-1 for all)", default=-1)

    args = parser.parse_args()
    return args

def evaluate_test(model, data, test_file_loc, output_file_loc, model_params, total=-1):
    """
    Evaluate and print metrics
    :param model: Model (use model.eval() to disable dropout / batchnorm)
    :param test_file_loc: testing file
    :param output_file_loc: output file
    :param model_params: params
    :param layers: default 3
    :return: None
    """
    layers = model_params['levels']
    test_file = pd.read_csv('../../data/' + test_file_loc)
    for i in range(model_params['levels']):
        test_file['pred_{}'.format(i)] = ''
        test_file['attn_{}'.format(i)] = ''
    test_file['recon_text'] = ''
    test_docs = []
    logging.info("Starting prediction ...")
    if total == -1:
        total = len(test_file)
    pb = tqdm(total=total)
    # pdb.set_trace()
    ct = 0
    attentions = []
    for i,row in test_file.iterrows():
        text = row['text']
        text = text.lower()
        text = data.tokenize(text)
        text = [data.word2id[w] if w in data.word2id else data.word2id[constants.UNK_WORD] for w in text]
        recon_text = [data.id2word[str(w)] for w in text]
        #print(text)
        #print(recon_text)
        text_len = len(text)
        src_text = Variable(torch.LongTensor([text]), volatile=True)
        src_len = [text_len]
        labels = [0]
        #pdb.set_trace()
        labels.extend([data.label2id['l{}_{}'.format(l,data.y_class2id['l'+str(l+1)][str(row['l{}'.format(l+1)])])]
                       for l in range(model_params['levels'])])
        labels = Variable(torch.LongTensor([labels]), volatile=True)
        if model_params['use_gpu']:
            src_text = src_text.cuda()
            labels = labels.cuda()
        renormalize = 'level'
        if 'renormalize' in model_params:
            renormalize = model_params['renormalize']
        loss, accs, attns, preds, correct, correct_confs, incorrect_confs = model.batchNLLLoss(
            src_text, src_len, labels,
            tf_ratio=0,
            loss_focus=model_params['loss_focus'],
            loss_weights=None,
            max_categories=3,
            target_level=1,
            attn_penalty_coeff=model_params['attn_penalty_coeff'],
            renormalize=renormalize)
        #print(preds)
        #print(correct_confs)
        #print(incorrect_confs)
        preds = [str(data.y_id2class['l'+str(indx+1)][int(data.id2label[p[0]].split('_')[1])]) for indx, p in enumerate(preds)]
        for idx, pred in enumerate(preds):
            test_file.at[i,'pred_{}'.format(idx)] = pred
        test_file.at[i,'recon_text'] = ' '.join(recon_text)
        ## Store attentions
        row_attentions = []
        for idx, attn in enumerate(attns):
            if type(attn) == list:
                attn = np.array([a.data.cpu().numpy() for a in attn])
                attn = np.squeeze(attn, axis=0)
            elif type(attn) == Variable:
                attn = attn.data.cpu().numpy()
                attn = np.squeeze(attn, axis=1)
            else:
                attn = []
            row_attentions.append(attn)
        pb.update(1)
        attentions.append(row_attentions)
        ct +=1
        if ct == total:
            break
    pb.close()
    # Calculate Metrics
    for layer in range(layers):
        test_file['l{}'.format(layer+1)] = test_file['l{}'.format(layer+1)].astype(str)
        test_file['pred_{}'.format(layer)] = test_file['pred_{}'.format(layer)].astype(str)
        acc = np.mean(test_file['l{}'.format(layer+1)] == test_file['pred_{}'.format(layer)])
        sk_acc = accuracy_score(test_file['l{}'.format(layer+1)], test_file['pred_{}'.format(layer)])
        sk_rec = recall_score(test_file['l{}'.format(layer+1)], test_file['pred_{}'.format(layer)], average='macro')
        sk_f1 = f1_score(test_file['l{}'.format(layer+1)], test_file['pred_{}'.format(layer)], average='macro')
        sk_precision = precision_score(test_file['l{}'.format(layer+1)], test_file['pred_{}'.format(layer)], average='macro')
        print("Layer {} Metrics".format(layer+1))
        print("Acc {}, Sk_accuracy {}, Recall {}, F1 Score {}, Precision {}".format(acc, sk_acc, sk_rec, sk_f1, sk_precision))

    logging.info("Done predicting. Saving file.")
    test_file.to_csv(output_file_loc)
    pkl.dump(attentions, open(output_file_loc + '_attentions.pkl', 'wb'))

if __name__ == '__main__':
    ## loading the model
    args = get_args()
    logging.info("Loading the model")
    model_params = json.load(open('../../saved/'+args.exp + '/parameters.json','r'))
    model = decoders.SimpleMLPDecoder(**model_params)
    # Load model
    model.load_state_dict(torch.load('../../saved/'+args.exp + '/' + args.model))
    logging.info("Loaded model")
    if model_params['use_gpu']:
        model = model.cuda()
    ## prepare the data
    logging.info("Loading the data")
    data = data_utils.Data_Utility(
        data_path=model_params['data_path'],
        train_test_split=model_params['train_test_split'],
        max_vocab=model_params['max_vocab'],
        max_word_doc=model_params['max_word_doc'],
        level = model_params['level'],
        decoder_ready=model_params['decoder_ready'],
        tokenization=model_params['tokenization'],
        clean=model_params['clean']
    )
    data.load(model_params['data_type'], model_params['data_loc'], model_params['file_name'])
    model.taxonomy = data.taxonomy

    model.eval()
    evaluate_test(model, data, args.file, args.output, model_params)









