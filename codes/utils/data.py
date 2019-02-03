## Data loader
## Read and prepare data for training.
## TODO: Use torch.DataLoader for efficient batch representations

import torch
import torch.utils.data as data
from torch.autograd import Variable
import os
from os.path import dirname, abspath
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import json
import random
import numpy as np
import pandas
from codes.utils import constants
from collections import Counter
import re
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from codes.utils.config import get_sample_config, get_config
from codes.utils.batch import Batch
import pdb
import pickle as pkl

base_loc = str(os.path.dirname(os.path.realpath(__file__)).split('codes')[0])

class Data_Utility(data.Dataset):
    def __init__(self, config):
        """
        Data Utility class
        """
        self.word2id = {}
        self.id2word = {}
        self.dyna_dict = {}
        self.train_indices = []
        self.test_indices = []
        self.split_ratio = config['train_test_split']
        self.special_tokens = [constants.PAD_WORD, constants.UNK_WORD]
        self.data_mode = 'train'
        self.decoder_ready = config['decoder_ready']
        self.max_vocab = config['max_vocab']
        self.max_word_doc = config['max_word_doc']
        self.level = config['level'] # select which levels to choose. if -1, then choose all
        self.levels = config['level'] # max number of levels to classify, default 3
        self.tokenization = config['tokenization']
        self.clean = config['clean']
        self.data_path = config['data_path']
        self.data_type = config['data_type']
        self.data_loc = config['data_loc']
        self.batch_size = config['batch_size']
        self.save_path_base = os.path.join(base_loc, 'data', self.data_path)
        self.save_loc = os.path.join(self.save_path_base,
                                     '{}_processed_{}.pkl'.format(self.data_type, self.tokenization))

        if not os.path.exists(self.save_path_base):
            os.makedirs(self.save_path_base)

    def preprocess(self):
        """
        Given data type and location, load and preprocess in an uniform format
        Also create dynamic dictionaries here
        :param data_type: WOS, LSHTC, WIKI
        :param data_loc: location of dataset
        :param tokenization: mode of tokenization : word, char
        :return:
        """
        ## Unified data format - should have two json files, one for data and other for ids / dictionaries
        data_m = {}
        dict_m = {}
        items = Counter() #set()
        y_classes = []
        text_data = []
        data_indexes = []
        y_class2id = {}
        full_data_csv_path = ''
        data_loc = os.path.join(base_loc,self.data_loc)

        if self.data_type == 'WOS':
            ## Web of science data
            logging.info("Reading WOS data")
            i = 0
            logging.info("Reading file X.txt")
            with open(os.path.join(data_loc, 'X.txt')) as fp:
                for line in fp:
                    text = self.tokenize(line.strip())
                    items.update(text)
                    ## prune docs by max words
                    if self.max_word_doc > 0 and len(text) > self.max_word_doc:
                        text = text[:self.max_word_doc]
                    text_data.append(text)
                    data_indexes.append(i)
                    i+=1
            logging.info("Read {} rows".format(i))
            dict_m['word2id'], dict_m['id2word'] = self.assign_wordids(items, self.special_tokens)
            ## add the level1, level2 and level3 in per array

            y_1 = []
            y_2 = []
            y_3 = []
            logging.info("Reading file YL1.txt")
            with open(os.path.join(data_loc, 'YL1.txt')) as fp:
                for line in fp:
                    y_1.append(int(line.strip()))
            logging.info("Reading file YL2.txt")
            with open(os.path.join(data_loc, 'YL2.txt')) as fp:
                for line in fp:
                    y_2.append(int(line.strip()))
            logging.info("Reading file Y.txt")
            with open(os.path.join(data_loc, 'Y.txt')) as fp:
                for line in fp:
                    y_3.append(int(line.strip()))
            for i in range(len(y_1)):
                y_classes.append([y_1[i],y_2[i],y_3[i]])
            # since classes are already in id format
            y_class2id = {'l1': {s:s for s in set(y_1)},
                          'l2': {s:s for s in set(y_2)},
                          'l3': {s:s for s in set(y_3)}}
            logging.info("Reading complete")


        elif self.data_type == 'WIKI':
            logging.info("Reading WIKI data")
            full_data_csv_path = os.path.join(data_loc,'wiki_data.csv')
            df = pandas.read_csv(full_data_csv_path)
            y_class2id = {'l1':{},'l2':{},'l3':{}}
            ct_dict = {'l1':0,'l2':0,'l3':0}

            def gen_class_id(row, level):
                class_name = row[level]
                ct = ct_dict[level]
                if class_name not in y_class2id[level]:
                    y_class2id[level][class_name] = ct
                    ct_dict[level] += 1
                return y_class2id[level][class_name]

            for i,row in df.iterrows():
                l_1 = gen_class_id(row,'l1')
                l_2 = gen_class_id(row, 'l2')
                l_3 = gen_class_id(row, 'l3')
                text = row['text']
                if not self.clean:
                    text = text.lower()
                if '<sent>' in text:
                    # data has been sentence tokenized
                    text = text.split('<sent>')
                    text = [self.tokenize(str(t)) for t in text]
                else:
                    text = self.tokenize(str(text))
                ## prune docs by max words
                if self.max_word_doc > 0 and len(text) > self.max_word_doc:
                    text = text[:self.max_word_doc]
                ## remove rows which do not have anything
                if len(text) == 0:
                    continue
                text_data.append(text)
                items.update(text)
                y_classes.append([l_1, l_2, l_3])
            logging.info("Read {} rows".format(len(text_data)))
            data_indexes = list(range(len(text_data)))

        data_m['y_class2id'] = y_class2id
        dict_m['word2id'], dict_m['id2word'] = self.assign_wordids(items, self.special_tokens)
        data_m['data_path'] = full_data_csv_path

        assert len(y_classes) == len(text_data)
        data_m['data'] = text_data
        data_m['labels'] = y_classes
        data_m['indexes'] = data_indexes


        ## create dynamic dictionary
        ## format: <l0>:[<l1>], <l1>:[<l2>]...
        ## append each node with its hierarchy label. l0_{node}
        logging.info("Creating dynamic dictionary")
        dy_dict = {}
        label_meta = {}
        for label_arr in data_m['labels']:
            for level, node in enumerate(label_arr):
                if level < len(label_arr) - 1:
                    node_rep = 'l{}_{}'.format(level, node)
                    if node_rep not in dy_dict:
                        dy_dict[node_rep] = []
                    dy_dict[node_rep].append(label_arr[level + 1])
                if level not in label_meta:
                    label_meta[level] = []
                label_meta[level].append(node)
        dict_m['dyna_dict'] = dy_dict
        dict_m['label_meta'] = label_meta

        s_labels = data_m['labels']
        # fix the labels. during data collection, the labels where taken as unique id per level.
        # to make all levels unique here for the decoder to work, we need to make them sequential
        label2id = {}
        # save a hierarchy of decoder levels
        taxonomy = {}
        logging.info("Building taxonomy...")
        # get the max number of labels per level
        max_levels = max([len(label) for label in s_labels])
        logging.info("Max number of labels per level : {}".format(max_levels))
        ct  = 1
        for i in range(max_levels):
            labels_in_level = set([label[i] for label in s_labels])
            for lb in labels_in_level:
                # add a special structure so it can be recovered later
                label2id['l{}_{}'.format(i,lb)] = ct
                ct +=1
        decoder_labels = []
        for labels in s_labels:
            row_labels = [0] # start with the go label
            for i, label in enumerate(labels):
                row_labels.append(label2id['l{}_{}'.format(i,label)])
            decoder_labels.append(row_labels)
        decoder_num_labels = ct
        # build the taxonomy
        taxonomy[0] = set()
        for labels in s_labels:
            new_dec_labels = [label2id['l{}_{}'.format(level,label)] for level,label in enumerate(labels)]
            parent = 0
            for dec_label in new_dec_labels:
                if parent not in taxonomy:
                    taxonomy[parent] = set()
                taxonomy[parent].add(dec_label)
                parent = dec_label
        data_m['decoder_labels'] = decoder_labels
        data_m['decoder_num_labels'] = decoder_num_labels
        data_m['taxonomy'] = taxonomy
        data_m['label2id'] = label2id
        data_m['id2label'] = {v:k for k,v in label2id.items()}
        logging.info("Done building taxonomy.")
        logging.info("Splitting train-test ...")
        self.labels = s_labels
        self.data_indexes = data_indexes
        self.split_indices()
        data_m['train_indices'] = self.train_indices
        data_m['test_indices'] = self.test_indices
        logging.info("Split done.")

        logging.info("Created. Saving files...")

        ## save the data and dictionaries
        pd = {
            'dict_m' : dict_m,
            'data_m' : data_m
        }
        save_path = os.path.join(self.save_path_base,
                                        '{}_processed_{}.pkl'.format(self.data_type, self.tokenization))
        pkl.dump(pd, open(save_path, 'wb'))
        logging.info("Saved in {}".format(save_path))
        return pd

    def get_level_labels(self, level=0):
        """
        return list of all labels in the particular level
        :return:
        """
        return list(set([p[level] for p in self.labels if len(p) >= (level + 1)]))

    def get_max_level(self):
        """
        return the deepest level
        :return:
        """
        return max([len(p) - 1 for p in self.labels])


    def tokenize(self, sent):
        """
        tokenize sentence based on mode
        :sent - sentence
        :param mode: word/char
        :return: splitted array
        """
        if self.clean:
            sent = text_cleaner(sent)
        if self.tokenization == 'word':
            return word_tokenize(sent)
        if self.tokenization == 'char':
            return sent.split()

    def assign_wordids(self, words, special_tokens=None):
        """
        Given a set of words, create word2id and id2word
        :param words: set of words
        :param special_tokens: set of special tokens to add into dictionary
        :return: word2id, id2word
        """
        count = 0
        word2id = {}
        ## if max_vocab is not -1, then shrink the word size
        if self.max_vocab >= 0:
            words = [tup[0] for tup in words.most_common(self.max_vocab)]
        else:
            words = list(words.keys())
        if special_tokens:
            for tok in special_tokens:
                word2id[tok] = count
                count +=1
        for word in words:
            word2id[word] = count
            count +=1
        id2word = {v:k for k,v in word2id.items()}
        return word2id, id2word

    def load(self):
        ## Load previously preprocessed data, and add to the object
        if not os.path.exists(self.save_loc):
            logging.info("Preprocessing...")
            processed_dict = self.preprocess()
        else:
            logging.info("Loading previously preprocessed data...")
            processed_dict = pkl.load(open(self.save_loc, 'rb'))
        self.word2id = processed_dict['dict_m']['word2id']
        self.id2word = processed_dict['dict_m']['id2word']
        self.dyna_dict = processed_dict['dict_m']['dyna_dict']
        self.data = processed_dict['data_m']['data']
        self.labels = processed_dict['data_m']['labels']
        self.label_meta = processed_dict['dict_m']['label_meta']
        self.data_indexes = processed_dict['data_m']['indexes']
        self.data_path = processed_dict['data_m']['data_path']
        self.y_class2id = processed_dict['data_m']['y_class2id']
        self.y_id2class = {cl:{v:k for k,v in dt.items()} for cl,dt in self.y_class2id.items()}
        #self.y_id2class = {v:k for k,v in self.y_class2id.items()} not possible as its a two level class
        self.taxonomy = processed_dict['data_m']['taxonomy']
        self.label2id = processed_dict['data_m']['label2id']
        self.id2label = processed_dict['data_m']['id2label']
        self.decoder_labels = processed_dict['data_m']['decoder_labels']
        self.decoder_num_labels = processed_dict['data_m']['decoder_num_labels']
        self.train_indices = processed_dict['data_m']['train_indices']
        self.test_indices = processed_dict['data_m']['test_indices']



    def split_indices(self):
        ## TODO: evenly split training and test so that test has atleast n examples of the categories
        end_labels = [labels[-1] for labels in self.labels]
        assert len(end_labels) == len(self.data_indexes)
        #print(len(end_labels))
        #print(len(set(end_labels)))
        label2rowid = {}
        for i,label in enumerate(end_labels):
            if label not in label2rowid:
                label2rowid[label] = []
            label2rowid[label].append(self.data_indexes[i])
        # shuffle within labelids
        for i,label in enumerate(end_labels):
            label2rowid[label] = random.sample(label2rowid[label], len(label2rowid[label]))
        train_indices = [v[:int(np.floor(len(v) * self.split_ratio))] for k,v in label2rowid.items()]
        train_indices = [v for k in train_indices for v in k]
        test_indices = [v[int(np.floor(len(v) * self.split_ratio)):] for k,v in label2rowid.items()]
        test_indices = [v for k in test_indices for v in k]
        # make sure no data bleeding has happened
        assert len(set(train_indices).intersection(set(test_indices))) == 0

        self.train_indices = random.sample(train_indices, len(train_indices))
        self.test_indices = random.sample(test_indices, len(test_indices))
        ## Split indices for train test
        ##all_indices = range(len(self.data))
        ##shuffled = random.sample(all_indices, len(all_indices))
        ##num_train = int(np.floor(len(all_indices) * self.split_ratio))
        ##self.train_indices = shuffled[:num_train]
        ##self.test_indices = shuffled[num_train:]

    def load_embedding(self,embedding_file='', embedding_saved='', embedding_dim=300, data_path=''):
        """
        Initialize the embedding from pre-trained vectors
        :param embedding_file: pre-trained vector file, eg glove.txt
        :param embedding_saved: file to save the embeddings
        :param embedding_dim: dimensions, eg 300
        :param data_path: data path
        :return: embedding matrix
        """

        embeddings = None
        word_dict = self.word2id

        parent_dir = dirname(dirname(dirname(abspath(__file__))))
        save_path_base = parent_dir + '/data/' + data_path
        if not os.path.exists(save_path_base):
            os.makedirs(save_path_base)
        emb_saved_full_path = save_path_base + embedding_saved

        if os.path.isfile(emb_saved_full_path):
            embeddings = torch.load(
                open(emb_saved_full_path, 'rb'))
        else:
            embeddings = torch.Tensor(len(word_dict), embedding_dim)
            embeddings.normal_(0, 1)
            word_count = 0

            # Fill in embeddings
            if not embedding_file:
                raise RuntimeError(
                    'Tried to load embeddings with no embedding file.')
            num_lines = sum(1 for line in open(embedding_file))
            pbar = tqdm(total=num_lines)
            done_words = []
            vec_word_dict = {}

            with open(embedding_file, 'r') as f:
                for line in f:
                    parsed = line.rstrip().split(' ')
                    if len(parsed) == 2:
                        # first line, so skip
                        pbar.update(1)
                        continue
                    assert (len(parsed) == embedding_dim + 1)
                    w = parsed[0]
                    if w in word_dict:
                        vec = [float(i) for i in parsed[1:]]
                        vec_word_dict[w] = vec
                        vec = torch.Tensor(vec)
                        embeddings[word_dict[w]].copy_(vec)
                        word_count += 1
                        done_words.append(w)
                    pbar.update(1)
            pbar.close()
            print('Done words : {}'.format(len(done_words)))
            # save the embeddings
            torch.save(embeddings, open(emb_saved_full_path, 'wb'))

        return embeddings


    def get_dataloader(self, mode='train'):
        ## return torch.DataLoader instance
        if mode == 'train':
            rows = self.train_indices
        else:
            rows = self.test_indices
        data_rows = []
        label_rows = []
        for row_index in rows:

            data = self.data[row_index]
            data = [self.word2id[word] if word in self.word2id else self.word2id[constants.UNK_WORD]
                    for word in data]
            labels = self.labels[row_index]
            if self.decoder_ready:
                labels = self.decoder_labels[row_index]
            if self.level != -1:
                labels = self.labels[row_index]
                labels = [0, labels[self.level]]
            data_rows.append(data)
            label_rows.append(labels)

        return torch.utils.data.DataLoader(TextDataLoader(data_rows, label_rows),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4)

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def calculate_weights(self, level=0):
        """
        Calculate class weights to correct for class imbalance
        :return:
        """
        if self.decoder_ready:
            all_labels = [lb[level] for lb in self.decoder_labels]
        else:
            all_labels = [lb[level] for lb in self.labels]
        label_count = Counter(all_labels)
        min_label_count = min(label_count.items(), key=lambda a: a[1])[1]
        label_weights = [min_label_count / count for label, count in sorted(label_count.items())]
        return label_weights


class TextDataLoader(data.Dataset):
    """
    Separate dataloader instance
    """
    def __init__(self, inp_rows, outp):
        self.inp_rows = inp_rows
        self.outp = outp

    def __getitem__(self, index):
        """
        Return single training row for dataloader
        :param item:
        :return:
        """
        inp_row = self.inp_rows[index]
        outp = self.outp[index]
        return inp_row, outp, [index]

    def __len__(self):
        return len(self.inp_rows)



### Helper function
def collate_fn(data):
    """
    helper function for torch.DataLoader
    :param data: list of tuples (data, labels)
    :return:
    """
    def merge(rows):
        lengths = [len(row) for row in rows]
        padded_rows = torch.zeros(len(rows), max(lengths)).long()
        for i, row in enumerate(rows):
            end = lengths[i]
            padded_rows[i,:end] = torch.LongTensor(row[:end])
        return padded_rows, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_data, src_labels, src_row_indexes = zip(*data)
    src_data, src_lengths = merge(src_data)
    src_labels = torch.LongTensor(src_labels)

    batch = Batch(src_data, src_labels, src_lengths, src_row_indexes)

    return batch

## Text cleaning function
def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


if __name__ == '__main__':
    config = get_config('7.dbp')
    ds = Data_Utility(config)
    ds.load()
    pdb.set_trace()
    dt = ds.get_dataloader(mode='train')
    for batch in dt:
        if min(batch.inp_lengths) <= 0:
            break
    dt = ds.get_dataloader(mode='test')
    for batch in dt:
        if min(batch.inp_lengths) <= 0:
            break
