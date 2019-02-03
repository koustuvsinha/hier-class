# Evaluate a saved model with respect to testing data
import torch
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
import argparse
from codes.models import decoders
from codes.utils import data as data_utils
from codes.utils import constants
import pdb
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from os.path import dirname, abspath

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

def predict(test_df, row_id, trainer, src_text, src_len, labels, data, mode='overall'):
    overall = True
    if mode != 'overall':
        overall = False
    loss, accs, attns, preds, correct, correct_confs, incorrect_confs, probs = trainer.batchNLLLoss(
        src_text, src_len, labels,
        mode='infer', overall=overall)
    # print(preds)
    # print(correct_confs)
    # print(incorrect_confs)
    preds = [str(data.y_id2class['l' + str(indx + 1)][int(data.id2label[p[0]].split('_')[1])]) for indx, p in
             enumerate(preds)]
    for idx, pred in enumerate(preds):
        test_df.at[row_id, 'pred_{}_{}'.format(mode, idx)] = pred
    return test_df, attns, probs

def calculate_metrics(layers, test_file, mode='overall'):
    print("Calculating metrics for mode : {}".format(mode))
    print("------------------------------------------------")
    for layer in range(layers):
        test_file['l{}'.format(layer+1)] = test_file['l{}'.format(layer+1)].astype(str)
        test_file['pred_{}_{}'.format(mode, layer)] = test_file['pred_{}_{}'.format(mode,layer)].astype(str)
        acc = np.mean(test_file['l{}'.format(layer+1)] == test_file['pred_{}_{}'.format(mode, layer)])
        sk_acc = accuracy_score(test_file['l{}'.format(layer+1)], test_file['pred_{}_{}'.format(mode, layer)])
        sk_rec = recall_score(test_file['l{}'.format(layer+1)], test_file['pred_{}_{}'.format(mode, layer)], average='macro')
        sk_f1 = f1_score(test_file['l{}'.format(layer+1)], test_file['pred_{}_{}'.format(mode, layer)], average='macro')
        sk_precision = precision_score(test_file['l{}'.format(layer+1)], test_file['pred_{}_{}'.format(mode, layer)], average='macro')
        print("Layer {} Metrics".format(layer+1))
        print("Acc {}, Sk_accuracy {}, Recall {}, F1 Score {}, Precision {}".format(acc, sk_acc, sk_rec, sk_f1, sk_precision))
    print('================================================')

def evaluate_test(trainer, data, test_file_loc, output_file_loc, model_params, total=-1):
    """
    Evaluate and print metrics
    :param model: Trainer (use trainer.model.eval() to disable dropout / batchnorm)
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
    probabilities = []
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
        test_file, attns_overall, probs_overall = predict(test_file, i, trainer, src_text, src_len, labels, data, mode='overall')
        test_file, attns_exact, probs_exact = predict(test_file, i, trainer, src_text, src_len, labels, data, mode='exact')
        test_file.at[i, 'recon_text'] = ' '.join(recon_text)
        ## Store attentions
        #row_attentions = []
        #pdb.set_trace()
        #for idx, attn in enumerate(attns):
        #    if type(attn) == list:
        #        attn = np.array([a.data.cpu().numpy() for a in attn])
        #        attn = np.squeeze(attn, axis=0)
        #    elif type(attn) == Variable:
        #        attn = attn.data.cpu().numpy()
        #        attn = np.squeeze(attn, axis=1)
        #    row_attentions.append(attn)
        pb.update(1)
        attentions.append([attns_overall, attns_exact])
        probabilities.append(probs_overall)
        ct +=1
        if ct == total:
            break
    pb.close()
    # Calculate Metrics
    calculate_metrics(layers, test_file, mode='overall')
    calculate_metrics(layers, test_file, mode='exact')

    ## store category embeddings
    """
    cat_emb = {}
    for cat_indx in range(sum(model_params['label_sizes']) + 1):
        cat_inp = Variable(torch.LongTensor([cat_indx]), volatile=True).cuda()
        cat_emb[cat_indx] = model.category_embedding(cat_inp).data.cpu().numpy()
        del cat_inp
    pkl.dump(cat_emb, open(output_file_loc + '_category_emb.pkl','wb'))
    pkl.dump(model.taxonomy, open(output_file_loc + '_taxonomy.pkl','wb'))
    """
    logging.info("Done predicting. Saving file.")
    test_file.to_csv(output_file_loc)
    pkl.dump(attentions, open(output_file_loc + '_attentions.pkl', 'wb'))
    pkl.dump(probabilities, open(output_file_loc + '_probs.pkl','wb'))

if __name__ == '__main__':
    ## loading the model
    args = get_args()
    logging.info("Loading the model")
    model_params = json.load(open('../../saved/'+args.exp + '/parameters.json','r'))
    ## load embeddings if any
    if model_params['use_embedding']:
        parent_dir = dirname(dirname(dirname(abspath(__file__))))
        save_path_base = parent_dir + '/data/' + model_params['data_path']
        model_params['embedding'] = torch.load(open(save_path_base + model_params['embedding_saved'], 'rb'))

    if model_params['model_type'] == 'attentive':
        model = decoders.AttentiveHierarchicalClassifier(**model_params)
    elif model_params['model_type'] == 'pooling':
        model = decoders.PooledHierarchicalClassifier(**model_params)
    #print(model_params)
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
    model_params['taxonomy'] = data.taxonomy
    trainer = decoders.Trainer(model=model, **model_params)

    trainer.model.eval()
    evaluate_test(trainer, data, args.file, args.output, model_params)









