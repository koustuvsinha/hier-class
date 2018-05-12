import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"

import numpy as np
from keras.models import Sequential
from Data_DBpedia import loadData_Tokenizer
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
    DATASET = 3
    np.set_printoptions(threshold=np.inf)
    '''
    location of input data in two ways 
    1: Tokenizer that is using GLOVE
    1: loadData that is using couting words or tf-idf
    '''

    # load the data
    d_train, df_test, word_index, embeddings_index =  \
            loadData_Tokenizer(DATASET, MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    print("Loading Data is Done")

    # initiate the models
    counter = 1 #keep track of number of models created
    print('Create model of RNN in level1')
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    model = BuildModel.buildModel_RNN(word_index, embedding_matrix,
                                      d_train.number_of_classes,MAX_SEQUENCE_LENGTH,
                                      EMBEDDING_DIM)

    HDLTex1 = []  # Level 2 models is list of Deep Structure
    HDLTex2=[[] for i in range(d_train.number_of_classes)]  # Level 3 models is list of list of Deep Structure
    # for i in range(0, d_train.number_of_classes):
    #     print('Create Sub model of level 1: ', i)
    #     HDLTex1.append(Sequential())
    #     HDLTex1[i] = BuildModel.buildModel_RNN(word_index, embedding_matrix,
    #                                           d_train.childs[i].number_of_classes,
    #                                           MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    #     counter +=1
    # #
    #
    # for i in range(0, d_train.number_of_classes):
    #     for j in range(0, d_train.childs[i].number_of_classes):
    #         print('Create Sub model of level 2:', i, j)
    #         HDLTex2[i].append(Sequential())
    #         HDLTex2[i][j] = BuildModel.buildModel_RNN(word_index, embedding_matrix,
    #                                                d_train.childs[i].childs[j].number_of_classes,
    #                                                MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    #         counter += 1

    print("%d number of models have been created"%counter)
    
    #initial model evaluation
    l1_acc,l2_acc,l3_acc = eval.evaluateDB(model,HDLTex1,HDLTex2,d_train,df_test)
    print("======test L1 acc:%f, L2 acc: %f on epoch %d=======" % (l1_acc,l2_acc,l3_acc, 0))

    #train the model combined with eval on test
    for epoch in range(epochs):
        print("starting to train on epoch %d"%epoch)
        model.fit(d_train.x_train, d_train.y_train,
              validation_data=(d_train.x_val, d_train.y_val),
              epochs=1,
              verbose=2,
              batch_size=batch_size_L1)

        for i in range(0, d_train.number_of_classes):
            HDLTex1[i].fit(d_train.childs[i].x_train, d_train.childs[i].y_train,
                          validation_data=(d_train.childs[i].x_val, d_train.childs[i].y_val),
                          epochs=1,
                          verbose=2,
                          batch_size=batch_size_L2)

        for i in range(0, d_train.number_of_classes):
            for j in range(d_train.childs[i].number_of_classes):
                HDLTex2[i][j].fit(d_train.childs[i].childs[j].x_train, d_train.childs[i].childs[j].y_train,
                               validation_data=(d_train.childs[i].childs[j].x_val, d_train.childs[i].childs[j].y_val),
                               epochs=1,
                               verbose=2,
                               batch_size=batch_size_L2)
        print("="*86)
        l1_acc, l2_acc, l3_acc = eval.evaluateDB(model, HDLTex1, HDLTex2, d_train, df_test)
        print("======test L1 acc:%f, L2 acc: %f on epoch %d=======" % (l1_acc, l2_acc, l3_acc, epoch))
        print("=" * 86)






