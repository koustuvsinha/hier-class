import re
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import WOS_input as WOS
import Download_Glove as GloVe
import numpy as np
import os
import pandas

''' Location of the dataset'''
path_WOS = WOS.download_and_extract()
GLOVE_DIR = GloVe.download_and_extract()


# print(GLOVE_DIR)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


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


class LabelDictionary():
    def __init__(self):
        self.y2i = {}
        self.i2y = {}


def create_class_dict(y_labels):
    dict = LabelDictionary()
    # this function takes labels in L2 and
    # transform into range (0,max_classes) for train
    unique_labels = set(np.unique(y_labels))
    for i, label in enumerate(unique_labels):
        dict.y2i[label] = i
        dict.i2y[i] = label
    return dict


def read(data_loc='',
         file_name='full_docs_2.csv', tokenization='word'):
    """
    Given data type and location, load data
    :param data_loc: location of dataset
    :param tokenization: mode of tokenization : word, char
    :return: (text,label) df_text is the tokenized text, df['l3'] last layer label
    """
    df = pandas.read_csv(data_loc + '/' + file_name)
    df = df.sample(frac=1).reset_index(drop=True)
    # df_texts = [self.tokenize(text) for text in df.text]
    # # df_texts = df.text.apply(self.tokenize)
    # # create dictionary
    # assert len(df_texts) == len(df['l3']) # l3 is the end level label
    # # print("finished tokenizing %d data instances"%len(df['l3']))
    # 
    # df = pandas.DataFrame(list(zip(df_texts, list(df['l3']))))
    # df.columns=['text', 'label']
    # # return df_texts, df['l3']
    return df


class Dataset():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None  # only the l1 data has test
        self.label_dict = None
        self.number_of_classes = None
        self.childens = None


class Dataset():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.dict = None
        self.number_of_classes = None
        self.childs = None


def data_pipline(df_train, level=1, stop_level=3):
    d = Dataset()
    d.number_of_classes = len(df_train['l%d' % level].unique())
    d.dict = create_class_dict(df_train['l%d' % level].unique())
    d.x_train, d.x_val, d.y_train, d.y_val = \
        train_test_split(df_train.text,
                         df_train['l%d' % level].apply(lambda x: d.dict.y2i[x]),
                         test_size=0.1, random_state=0)

    children_data = [None] * len(d.dict.y2i)
    next_level = 1 + level
    if stop_level + 1 == next_level:
        d_childs = None
    else:
        for i in range(len(d.dict.i2y.keys())):
            children_data[i] = data_pipline(df_train[df_train['l%d' % level] == d.dict.i2y[i]],
                                            level=next_level)
        d.childs = children_data
    return d


def loadData_Tokenizer(DATASET, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    # every dataset has x_train, y_train, x_val, y_val, label_dict, number_of_classes, list of children_data
    # except the first level dataset also have test
    data_loc = '/home/ml/ksinha4/mlp/hier-class/data'
    df_train = read(data_loc=data_loc, file_name="df_small_train.csv")
    df_test = read(data_loc=data_loc, file_name="df_small_test.csv")
    df_train.text = [clean_str(x) for x in df_train.text]
    df_test.text = [clean_str(x) for x in df_test.text]

    print('Start tokenization.')
    x_train = df_train.text.values
    x_test = df_test.text.values

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_pad_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    x_train_seq = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test_seq = pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
    df_train['text'] = x_train_seq.tolist()
    df_test['text'] = x_test_seq.tolist()

    print("content shape", df_train.text.shape)


    d_train = data_pipline(df_train, level=1, stop_level=3)
    
    embeddings_index = {}
    '''
    For CNN and RNN, we used the text vector-space models using $100$ dimensions as described in Glove. A vector-space model is a mathematical mapping of the word space
    '''
    Glove_path = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    # print(Glove_path)
    f = open(Glove_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print("Warnning" + str(values) + " in" + str(line))
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (d_train, df_test, word_index, embeddings_index)


def loadData():
    WOS.download_and_extract()
    fname = os.path.join(path_WOS, "WebOfScience/WOS5736/X.txt")
    fnamek = os.path.join(path_WOS, "WebOfScience/WOS5736/YL1.txt")
    fnameL2 = os.path.join(path_WOS, "WebOfScience/WOS5736/YL2.txt")
    with open(fname) as f:
        content = f.readlines()
        content = [text_cleaner(x) for x in content]
    with open(fnamek) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    with open(fnameL2) as fk:
        contentL2 = fk.readlines()
        contentL2 = [x.strip() for x in contentL2]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)
    number_of_classes_L1 = np.max(Label) + 1  # number of classes in Level 1

    Label_L2 = np.matrix(contentL2, dtype=int)
    Label_L2 = np.transpose(Label_L2)
    np.random.seed(7)
    print(Label.shape)
    print(Label_L2.shape)
    Label = np.column_stack((Label, Label_L2))

    number_of_classes_L2 = np.zeros(number_of_classes_L1, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=0)

    vectorizer_x = CountVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()

    L2_Train = []
    L2_Test = []
    content_L2_Train = []
    content_L2_Test = []

    for i in range(0, number_of_classes_L1):
        L2_Train.append([])
        L2_Test.append([])
        content_L2_Train.append([])
        content_L2_Test.append([])

    for i in range(0, X_train.shape[0]):
        L2_Train[y_train[i, 0]].append(y_train[i, 1])
        number_of_classes_L2[y_train[i, 0]] = max(number_of_classes_L2[y_train[i, 0]], (y_train[i, 1] + 1))
        content_L2_Train[y_train[i, 0]].append(X_train[i])

    for i in range(0, X_test.shape[0]):
        L2_Test[y_test[i, 0]].append(y_test[i, 1])
        content_L2_Test[y_test[i, 0]].append(X_test[i])

    for i in range(0, number_of_classes_L1):
        L2_Train[i] = np.array(L2_Train[i])
        L2_Test[i] = np.array(L2_Test[i])
        content_L2_Train[i] = np.array(content_L2_Train[i])
        content_L2_Test[i] = np.array(content_L2_Test[i])
    return (
        X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2)
