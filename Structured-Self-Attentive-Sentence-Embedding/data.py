## Data loader
## Read and prepare data for training.

import torch
import torch.utils.data as data
from torch.autograd import Variable
from collections import Counter
import os
from os.path import dirname, abspath
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import json
import random
import numpy as np
import pandas
from hier_class.utils import constants
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Data_Utility(data.Dataset):
    def __init__(self, exp_name='', train_test_split=0.8,
                 decoder_ready=False, max_vocab=-1, max_word_doc=-1):
        """

        :param exp_name:
        :param train_test_split:
        :param decoder_ready: If decoder_ready = True, then return labels with a starting 0 label
         and shift all labels by 1
        """
        self.word2id = {}
        self.id2word = {}
        self.dyna_dict = {}
        self.cat2id={}
        self.id2cat={}
        self.train_indices = []
        self.test_indices = []
        self.split_ratio = train_test_split
        self.special_tokens = [constants.PAD_WORD, constants.UNK_WORD]
        self.data_mode = 'train'
        self.decoder_ready = decoder_ready
        self.max_vocab = max_vocab
        self.max_word_doc = max_word_doc
        parent_dir = dirname(dirname(dirname(abspath(__file__))))
        self.save_path_base = parent_dir + '/saved/' + exp_name
        if not os.path.exists(self.save_path_base):
            os.makedirs(self.save_path_base)

    def read(self,  data_loc='',
            file_name='full_docs_2.csv', tokenization='word'):
        """
        Given data type and location, load data
        :param data_loc: location of dataset
        :param tokenization: mode of tokenization : word, char
        :return: (text,label) df_text is the tokenized text, df['l3'] last layer label
        """
        df = pandas.read_csv(data_loc + '/' + file_name)
        df = df.sample(frac=1).reset_index(drop=True)
        # print(df.head(10))
        # print("finished loading %d data instances" % len(df))
        df_texts = [self.tokenize(text) for text in df.text]
        # df_texts = df.text.apply(self.tokenize)
        # create dictionary
        assert len(df_texts) == len(df['l3']) # l3 is the end level label
        # print("finished tokenizing %d data instances"%len(df['l3']))

        df = pandas.DataFrame(list(zip(df_texts, list(df['l3']))))
        df.columns=['text', 'label']
        # return df_texts, df['l3']
        return df

    def assign_category_ids(self, y):
        id = 0
        cat2id = {}
        category_set = list(set(y))
        for key in category_set:
            cat2id[key] = id
            id += 1
        id2cat = {v: k for k, v in cat2id.items()}
        self.cat2id=cat2id
        self.id2cat=id2cat


    def add_to_cat2id(self,key):
        id = len(self.cat2id)
        self.cat2id[key] = id
        self.id2cat[id]=key

    def transfer_cat_to_id(self, df_y):
        """
        this method transfers list of labels into list of category ids
        :param df_y: list of texts
        :param c2i: dictionary for category to ids
        :return: df_ids: list of categories in their ids
        """
        output=[]
        for w in df_y:
            if w not in self.cat2id.keys():
                self.add_to_cat2id(w)
        y_in_id = [int(self.cat2id[w]) for w in df_y]
        assert max(y_in_id) < len(self.cat2id)
        return y_in_id


    def assign_word_ids(self, df_texts,
                        special_tokens=["<pad>", "<unk>","<sos>","<eos>"],
                        vocab_size=-1):
        """
        Given df_texts (list of sent tokens), create word2id and id2word
        based on the most common words
        :param  df_text: list of sent tokens
        :param special_tokens: set of special tokens to add into dictionary
        :param vocab_size: max_number of vocabs
        :return: word2id, id2word
        """
        id = 0
        word2id = {}
        # add special tokens in w2i
        for tok in special_tokens:
            word2id[tok] = id
            id += 1
            print(tok,word2id[tok], end=' ')

        word_set = [element for text in df_texts for element in text]
        c = Counter(word_set)

        ## if max_vocab is not -1, then shrink the word size
        if vocab_size >= 0:
            words = [tup[0] for tup in c.most_common(vocab_size - len(special_tokens))]
        else:
            words = list(c.keys())

        # add regular words in
        for word in words:
            word2id[word] = id
            id += 1
        id2word = {v: k for k, v in word2id.items()}
        # print('finishing processing %d vocabs' % len(word2id))
        return word2id, id2word

    def transfer_word_to_id(self, df_texts, w2i):
        """
        this method transfers list of text into list of w2i ids
        :param df_texts: list of texts
        :param w2i: dictionary for word to ids
        :return: df_ids: list of texts in their ids
        """
        dict_len = len(w2i)
        def transfer_word_2_id(text,w2i_dict):
            text_in_id = [w2i_dict[w] if w in w2i_dict.keys()
             else w2i_dict["<unk>"] for w in text]
            assert max(text_in_id) < dict_len

        df_ids = [transfer_word_2_id(text, w2i) for text in df_texts]
        return df_ids

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


    def tokenize(self, sent, mode='word'):
        """
        tokenize sentence based on mode
        :sent - sentence
        :param mode: word/char
        :return: splitted array
        """
        if mode == 'word':
            return word_tokenize(sent)
        if mode == 'char':
            return sent.split()



    def load(self, data_type='', data_loc='', file_name='', tokenization='word'):
        ## Load previously preprocessed data, and add to the object
        save_loc = self.save_path_base + '/{}_processed_{}.json'.format(data_type, tokenization)
        if not os.path.exists(save_loc):
            logging.info("Preprocessing...")
            processed_dict = self.preprocess(data_type, data_loc, file_name, tokenization)
        else:
            logging.info("Loading previously preprocessed data...")
            processed_dict = json.load(open(save_loc))
        self.word2id = processed_dict['dict_m']['word2id']
        self.id2word = processed_dict['dict_m']['id2word']
        self.dyna_dict = processed_dict['dict_m']['dyna_dict']
        self.data = processed_dict['data_m']['data']
        self.labels = processed_dict['data_m']['labels']
        self.label_meta = processed_dict['dict_m']['label_meta']
        if self.decoder_ready:
            # fix the labels. during data collection, the labels where taken as unique id per level.
            # to make all levels unique here for the decoder to work, we need to make them sequential
            label2id = {}
            # get the max number of labels per level
            max_levels = max([len(label) for label in self.labels])
            ct  = 1
            for i in range(max_levels):
                labels_in_level = set([label[i] for label in self.labels])
                for lb in labels_in_level:
                    # add a special structure so it can be recovered later
                    label2id['l{}_{}'.format(i,lb)] = ct
                    ct +=1
            self.decoder_labels = []
            for labels in self.labels:
                row_labels = [0] # start with the go label
                for i, label in enumerate(labels):
                    row_labels.append(label2id['l{}_{}'.format(i,label)])
                self.decoder_labels.append(row_labels)
            self.decoder_num_labels = ct

        self.split_indices()

    def split_indices(self):
        ## Split indices for train test
        all_indices = range(len(self.data))
        shuffled = random.sample(all_indices, len(all_indices))
        num_train = int(np.floor(len(all_indices) * self.split_ratio))
        self.train_indices = shuffled[:num_train]
        self.test_indices = shuffled[num_train:]

    def load_embedding(self,embedding_file='', embedding_saved='', embedding_dim=300, exp_name=''):
        """
        Initialize the embedding from pre-trained vectors
        :param embedding_file: pre-trained vector file, eg glove.txt
        :param embedding_saved: file to save the embeddings
        :param embedding_dim: dimensions, eg 300
        :param exp_name: experiment name
        :return: embedding matrix
        """

        embeddings = None
        word_dict = self.word2id

        parent_dir = dirname(dirname(dirname(abspath(__file__))))
        save_path_base = parent_dir + '/saved/' + exp_name
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
            # save the embeddings
            torch.save(embeddings, open(emb_saved_full_path, 'wb'))

        return embeddings


    def save(self):
        ## save preprocessed entities
        pass

    def __getitem__(self, index):
        ## return single training row for torch.DataLoader
        if self.data_mode == 'train':
            rows = self.train_indices
        else:
            rows = self.test_indices
        data = self.data[rows[index]]

        data = [self.word2id[word] for word in data]

        # data = [self.word2id[word] for word in data]
        #labels = self.labels[rows[index]]

        if type(data[0]) != list:
            data = [[self.word2id[word] if word in self.word2id
                else self.word2id[constants.UNK_WORD]
                for word in data]]
        else:
            num_sent = len(data)
            sents_len = [len(sent) for sent in data]
            max_sent_len = max(sents_len)
            matrix = np.zeros((num_sent, max_sent_len))

            for i, sent in enumerate(data):
                matrix[i][:sents_len[i]] = [self.word2id[word] if word in self.word2id
                                            else self.word2id[constants.UNK_WORD]
                                            for word in sent]
            # data = [self.word2id[word] for sent in data for word in sent]
            data = matrix

        labels = self.labels[rows[index]]
        if self.decoder_ready:
            labels = self.decoder_labels[rows[index]]
        data = torch.LongTensor(data)
        return data, labels

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.train_indices)
        else:
            return len(self.test_indices)

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
            padded_rows[i,:end] = row[:end]
        return padded_rows, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_data, src_labels = zip(*data)
    src_data, src_lengths = merge(src_data)

    return src_data, src_lengths, src_labels

