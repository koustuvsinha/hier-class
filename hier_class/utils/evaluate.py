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

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

## arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", type=str, help="experiment to load", default="wiki_normalize_experiment2018-04-21_18:43:17")
parser.add_argument("-m","--model", type=str, help="model to load", default="model_epoch_0_step_0.mod")
parser.add_argument("-f","--file", type=str, help="testing file to load", default="full_docs_2_test.csv")
parser.add_argument("-o","--output", type=str, help="file to write the output", default="output.csv")
parser.add_argument("-c","--confidence", type=float, help="confidence to measure pruned accuracy", default=0.0)
parser.add_argument("-n","--num", type=int, help="number of evals (-1 for all)", default=-1)

args = parser.parse_args()

if __name__ == '__main__':
    ## loading the model
    logging.info("Loading the model")
    model_params = json.load(open('../../saved/'+args.exp + '/parameters.json','r'))
    model = decoders.SimpleMLPDecoder(**model_params)
    # TODO: load model!!!!!!!!!!
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
        decoder_ready=model_params['decoder_ready']
    )
    data.load(model_params['data_type'], model_params['data_loc'], model_params['file_name'], model_params['tokenization'])
    model.taxonomy = data.taxonomy
    test_file = pd.read_csv('../../data/' + args.file)
    test_docs = []
    logging.info("Starting prediction ...")
    total = args.num
    if total == -1:
        total = len(test_file)

    ## TODO: calculate decoder labels

    pb = tqdm(total=total)
    #pdb.set_trace()
    ct = 0
    for i,row in test_file.iterrows():
        text = row['text']
        text = text.lower()
        text = data.tokenize(text,model_params['tokenization'])
        text = [data.word2id[w] if w in data.word2id else data.word2id[constants.UNK_WORD] for w in text]
        text_len = len(text)
        src_text = Variable(torch.LongTensor([text]), volatile=True)
        src_len = [text_len]
        labels = [0]
        #pdb.set_trace()
        labels.extend([data.label2id['l{}_{}'.format(l,data.y_class2id['l'+str(l+1)][row['l{}'.format(l+1)]])]
                       for l in range(model_params['levels'])])
        labels = Variable(torch.LongTensor([labels]), volatile=True)
        if model_params['use_gpu']:
            src_text = src_text.cuda()
            labels = labels.cuda()
        loss, accs, attns, preds, correct, correct_confs, incorrect_confs = model.batchNLLLoss(
            src_text, src_len, labels,
            tf_ratio=0,
            loss_focus=model_params['loss_focus'],
            loss_weights=None,
            max_categories=3,
            target_level=1,
            attn_penalty_coeff=model_params['attn_penalty_coeff'],
            renormalize=model_params['renormalize'])
        #print(preds)
        #print(correct_confs)
        #print(incorrect_confs)
        preds = [data.y_id2class['l'+str(indx+1)][int(data.id2label[p[0]].split('_')[1])] for indx, p in enumerate(preds)]
        for idx, pred in enumerate(preds):
            test_file.set_value(i,'pred_{}'.format(idx), pred)
        pb.update(1)
        ct +=1
        if ct == total:
            break

    pb.close()

    logging.info("Done predicting. Saving file.")
    test_file.to_csv(args.output)








