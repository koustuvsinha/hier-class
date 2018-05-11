import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"

import numpy as np
from keras.models import Sequential
import Data_DBpedia
import BuildModel
import eval

if __name__ == "__main__":

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 500 # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 55000 # Maximum number of unique words
    EMBEDDING_DIM = 100 #embedding dimension you can change it to {25, 100, 150, and 300} but need to change glove version
    batch_size_L1 = 64 # batch size in Level 1
    batch_size_L2 = 64 # batch size in Level 2
    epochs = 40
    L1_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
    L2_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2
    DATASET = 2
    np.set_printoptions(threshold=np.inf)
    '''
    location of input data in two ways 
    1: Tokenizer that is using GLOVE
    1: loadData that is using couting words or tf-idf
    '''

    # load the data
    X_train, y_train, X_val, y_val,X_test, y_test, \
    content_L2_Train, L2_Train, content_L2_Val, L2_Val,\
    number_of_classes_L2,L2_dict_y2i,\
    word_index, embeddings_index,number_of_classes_L1 =  \
            Data_DBpedia.loadData_Tokenizer(DATASET, MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    L2_dict_i2y = [dict((v, k) for k, v in d.items()) for d in L2_dict_y2i]
    print("Loading Data is Done")

    # initiate the models
    print('Create model of RNN in level1')
    model = BuildModel.buildModel_RNN(word_index, embeddings_index,
                                      number_of_classes_L1,MAX_SEQUENCE_LENGTH,
                                      EMBEDDING_DIM)
    HDLTex = []  # Level 2 models is list of Deep Structure
    for i in range(0, number_of_classes_L1):
        print('Create Sub model of ', i)
        HDLTex.append(Sequential())
        HDLTex[i] = BuildModel.buildModel_RNN(word_index, embeddings_index,
                                              number_of_classes_L2[i],
                                              MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    #initial model evaluation
    l1_acc,l2_acc = eval.evaluate(model, HDLTex, L2_dict_i2y, X_test, y_test)
    print("======test L1 acc:%f, L2 acc: %f on epoch %d=======" % (l1_acc,l2_acc , 0))

    #train the model combined with eval on test
    for epoch in range(epochs):
        print("starting to train on epoch %d"%epoch)
        model.fit(X_train, y_train[:, 0],
              validation_data=(X_val, y_val[:, 0]),
              epochs=1,
              verbose=2,
              batch_size=batch_size_L1)

        for i in range(0, number_of_classes_L1):
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Val[i], L2_Val[i]),
                          epochs=1,
                          verbose=2,
                          batch_size=batch_size_L2)
        print("="*86)
        l1_acc, l2_acc = eval.evaluate(model, HDLTex, L2_dict_i2y, X_test, y_test)
        print("======test L1 acc:%.4f, L2 acc: %.4f on epoch %d=======" % (l1_acc, l2_acc, epoch))
        print("=" * 86)






