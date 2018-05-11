# Please create your own dataloader for new datasets of the following type

import torch, keras
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils
from data import Data_Utility
from util import Dictionary
import pickle
import os.path


def load_data_set(vocab_size,dataset_type):
    """
        Loads the dataset.
        Args:
            dataset_type: Dbpedia or WIKI
			vocab_size: {int} size of the vocabulary
        Returns:
            x_train: {df} with col_names=['text','label']
            x_val: {df} df['text'][0] is an np array with indices of words
            x_test: {df} df['label'][0] is an np.int with indice of the label category
            word_to_id  : {dict} words mapped to indices
            cat2id: {dict} categories mapped to indices
        """

    save_list_str = ["x_train", "x_val", "x_test", "dictionary", "dataLoader.cat2id"]
    return_list = []
    if os.path.exists("./data/%s.x_train.p"%dataset_type):
        print("---loading pre-process %s data---"%dataset_type)
        # load already processed data
        for item in save_list_str:
            with open("./data/%s."%dataset_type+item+".p","rb") as f:
                 return_list.append(pickle.load(f))
        print("----finish data loading----{%d} train,{%d} val, {%d} test"
              % (len(return_list[0]), len(return_list[1]), len(return_list[2])))
        print("{%d} words in dictionary, {%d} classes----"
              % (len(return_list[3]), len(return_list[4])))
        return return_list

    print("----initial %s data loading and processing----"%dataset_type)
    dataLoader = Data_Utility()
    if dataset_type == "DBpedia":
        data_loc = '/home/ml/ksinha4/mlp/hier-class/data'
        x_train = dataLoader.read(data_loc=data_loc, file_name="df_small_train.csv")
        x_test = dataLoader.read(data_loc=data_loc, file_name="df_small_test.csv")
    elif dataset_type == "WIKI":
        data_loc = '/home/ml/ksinha4/datasets/data_WIKI'
        x_train = dataLoader.read(data_loc=data_loc, file_name="full_docs_2_train.csv")
        x_test = dataLoader.read(data_loc=data_loc, file_name="full_docs_2_test.csv")
        # "/home/ml/ksinha4/datasets/data_WOS/WebOfScience/WOS46985"
    else:
        raise Exception('this dataset type is not implemented yet')
    x_val = x_train[:int(0.1 * len(x_train))]
    x_train = x_train[int(0.1 * len(x_train)):]
    print("----finish data loading----{%d} train,{%d} val, {%d} test"
          % (len(x_train), len(x_val), len(x_test)))

    # processing dictionary and cat2id
    dictionary = Dictionary()
    dictionary.word2idx, dictionary.idx2word = dataLoader.assign_word_ids(
        x_train['text'].append(x_val['text']), vocab_size=vocab_size)
    dataLoader.assign_category_ids(list(x_train['label']) + \
                                   list(x_val['label']) + list(x_test['label']))
    x_train['label'] = dataLoader.transfer_cat_to_id(x_train['label'])
    x_val['label'] = dataLoader.transfer_cat_to_id(x_val['label'])
    x_test['label'] = dataLoader.transfer_cat_to_id(x_test['label'])
    print("----processed {%d} word_2_id, {%d}cat_2_id----" %\
          (len(dictionary.word2idx),len(dataLoader.cat2id)))

    # save the processed files in pickle
    save_list=[x_train, x_val, x_test, dictionary, dataLoader.cat2id]
    for i in range(len(save_list_str)):
        if not os.path.exists("./data"):
            os.mkdir("./data")
        with open("./data/%s.%s.p"%(dataset_type,save_list_str[i]),'wb') as f:
            pickle.dump(save_list[i],f)
    return x_train, x_val, x_test, dictionary, dataLoader.cat2id