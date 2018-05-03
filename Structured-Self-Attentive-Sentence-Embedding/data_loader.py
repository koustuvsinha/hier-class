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


def load_data_set(vocab_size):
    """
        Loads the dataset.
        Args:
            type: 1 for multiclass classfication return wiki set
            max_len: {int} timesteps used for padding
			vocab_size: {int} size of the vocabulary
			batch_size: batch_size
        Returns:
            train_loader: {torch.Dataloader} train dataloader
            x_test_pad  : padded tokenized test_data for cross validating
			y_test      : y_test
            word_to_id  : {dict} words mapped to indices
 
      
        """
    INDEX_FROM = 3
    # wiki
    # data_loc = '/home/ml/ksinha4/datasets/data_WIKI'
    # print("----initial data loading----")
    # dataLoader = Data_Utility()
    # x_train_texts, y_train = dataLoader.read(data_loc=data_loc, file_name="full_docs_2_train.csv")
    # x_val_texts, y_val = x_train_texts[:int(0.1 * len(y_train))], y_train[:int(0.1 * len(y_train))]
    # x_train_texts, y_train = x_train_texts[int(0.1 * len(y_train)):], y_train[int(0.1 * len(y_train)):]
    # x_test_texts, y_test = dataLoader.read(data_loc=data_loc, file_name="full_docs_2_test.csv")
    # print("----finish data loading----{%d} train,{%d} val, {%d} test"
    #       % (len(y_train), len(y_val), len(y_test)))

    # DB pedia

    save_list = ["x_train", "x_val", "x_test", "dictionary", "dataLoader.cat2id"]
    if os.path.exists("x_train"):

    data_loc = '/home/ml/ksinha4/mlp/hier-class/data'
    print("----initial data loading----")
    dataLoader = Data_Utility()
    dictionary = Dictionary()
    x_train = dataLoader.read(data_loc=data_loc, file_name="df_small_train.csv")
    x_val = x_train[:int(0.1 * len(x_train))]
    x_train = x_train[int(0.1 * len(x_train)):]
    x_test = dataLoader.read(data_loc=data_loc, file_name="df_small_test.csv")
    print("----finish data loading----{%d} train,{%d} val, {%d} test"
          % (len(x_train), len(x_val), len(x_test)))

    dictionary.word2idx, dictionary.idx2word = dataLoader.assign_word_ids(
        x_train['text'].append(x_val['text']), vocab_size=vocab_size)
    dataLoader.assign_category_ids(list(x_train['label']) + list(x_val['label']) + list(x_test['label']))
    x_train['label'] = np.array(dataLoader.transfer_cat_to_id(x_train['label']))
    x_val['label'] = np.array(dataLoader.transfer_cat_to_id(x_val['label']))
    x_test['label'] = np.array(dataLoader.transfer_cat_to_id(x_test['label']))
    print("----processed {%d} word_2_id, {%d}cat_2_id----" % (len(dictionary.word2idx),
                                                              len(dataLoader.cat2id)))
    save_list=[x_train, x_val, x_test, dictionary, dataLoader.cat2id]
    for item in save_list:
        with open("%s.p"%item,'wb') as f:
            pickle.dump(item,f)
    return x_train, x_val, x_test, dictionary, dataLoader.cat2id
    # x_train = dataLoader.transfer_word_to_id(x_train_texts, w2i)
    # x_val = dataLoader.transfer_word_to_id(x_val_texts, w2i)
    # x_test = dataLoader.transfer_word_to_id(x_test_texts, w2i)

    # print("----finish transfering w2i,c2i----")
    #
    # x_train_pad = pad_sequences(x_train, maxlen=max_len, padding='post')
    # x_val_pad = pad_sequences(x_val, maxlen=max_len, padding='post')
    # x_test_pad = pad_sequences(x_test, maxlen=max_len, padding='post')
    #
    #
    # train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),
    #                                       torch.from_numpy(y_train).type(torch.LongTensor))
    # val_data = data_utils.TensorDataset(torch.from_numpy(x_val_pad).type(torch.LongTensor),
    #                                       torch.from_numpy(y_val).type(torch.LongTensor))
    # test_data = data_utils.TensorDataset(torch.from_numpy(x_test_pad).type(torch.LongTensor),
    #                                       torch.from_numpy(y_test).type(torch.LongTensor))
    #
    # train_loader = data_utils.DataLoader(train_data, batch_size=batch_size,
    #                                      shuffle=True, drop_last=True)
    # val_loader = data_utils.DataLoader(val_data, batch_size=batch_size,
    #                                      shuffle=True, drop_last=True)
    # test_loader = data_utils.DataLoader(test_data, batch_size=batch_size,
    #                                     shuffle=True, drop_last=True)
    #
    # return train_loader, val_loader, test_loader, w2i, dataLoader.cat2id
