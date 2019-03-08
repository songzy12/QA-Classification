# coding: utf-8

import io
import json
import pandas as pd
from collections import Counter
from tflearn.data_utils import pad_sequences
import random
import numpy as np
import h5py
import pickle
print("import package successful...")

# read source file as csv
base_path = './data/'


def get_data_x_y(filename):
    with io.open(filename, encoding='utf8') as f:
        m = json.loads(f.read())

    train_data_x = []
    train_data_y = []
    for k, v in m.items():
        train_data_x.append(k)
        train_data_y.append(v)
    return train_data_x, train_data_y


def tokenize(text):
    return list(text)


train_data_x, train_data_y = get_data_x_y(base_path + 'svm/train.json')

# create vocabulary_dict, label_dict, generate training/validation data, and save to some place

# create vocabulary of charactor token by read word_embedding.txt
word_embedding_object = open(base_path+'merge_sgns_bigram_char300.txt')
lines_wv = word_embedding_object.readlines()
word_embedding_object.close()
char_list = []
char_list.extend(['PAD', 'UNK', 'CLS', 'SEP', 'unused1',
                  'unused2', 'unused3', 'unused4', 'unused5'])
PAD_ID = 0
UNK_ID = 1
for i, line in enumerate(lines_wv):
    if i == 0:
        continue
    char_embedding_list = line.split(" ")
    char_token = char_embedding_list[0]
    char_list.append(char_token)

# write to vocab.txt under data/ieee_zhihu_cup
vocab_path = base_path+'vocab.txt'
vocab_char_object = open(vocab_path, 'w')

word2index = {}
for i, char in enumerate(char_list):
    if i < 10:
        print(i, char)
    word2index[char] = i
    vocab_char_object.write(char+"\n")
vocab_char_object.close()
print("vocabulary of char generated....")


# generate labels list, and save to file system
c_labels = Counter()
train_data_y_small = train_data_y[0:100000]  # .sample(frac=0.1)
for index, topic_ids in enumerate(train_data_y_small):
    topic_list = [topic_ids]
    c_labels.update(topic_list)

label_list = c_labels.most_common()
label2index = {}
label_target_object = open(base_path+'label_set.txt', 'w')
for i, label_freq in enumerate(label_list):
    label, freq = label_freq
    label2index[label] = i
    label_target_object.write(label+"\n")
    if i < 20:
        print(label, freq)
label_target_object.close()
print("generate label dict successful...")


def transform_multilabel_as_multihot(label_list, label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1
    return result


def get_X_Y(train_data_x, train_data_y, label_size, test_mode=False):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """
    X = []
    Y = []
    if test_mode:
        # todo todo todo todo todo todo todo todo todo todo todo todo
        train_data_x_tiny_test = train_data_x[0:1000]
        # todo todo todo todo todo todo todo todo todo todo todo todo
        train_data_y_tiny_test = train_data_y[0:1000]
    else:
        train_data_x_tiny_test = train_data_x
        train_data_y_tiny_test = train_data_y

    for index, title_char in enumerate(train_data_x_tiny_test):
        # split into list
        title_char_list = tokenize(title_char)
        # transform to indices
        title_char_id_list = [word2index.get(
            x, UNK_ID) for x in title_char_list if x.strip()]

        X.append(title_char_id_list)
        if index < 3:
            print(index, title_char_id_list)
        if index % 100000 == 0:
            print(index, title_char_id_list)

    for index, topic_ids in enumerate(train_data_y_tiny_test):
        topic_id_list = [topic_ids]
        label_list_dense = [label2index[l] for l in topic_id_list if l.strip()]
        label_list_sparse = transform_multilabel_as_multihot(
            label_list_dense, label_size)
        Y.append(label_list_sparse)
        if index % 100000 == 0:
            print(index, ";label_list_dense:", label_list_dense)

    return X, Y


def save_data(cache_file_h5py, cache_file_pickle, word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index, label2index), target_file)


# generate training/validation/test data using source file and vocabulary/label set.
#  get X,Y---> shuffle and split data----> save to file system.
test_mode = False
label_size = len(label2index)
cache_path_h5py = base_path+'data.h5'
cache_path_pickle = base_path+'vocab_label.pik'
max_sentence_length = 200

# step 1: get (X,y)
train_X, train_Y = get_X_Y(train_data_x, train_data_y,
                           label_size, test_mode=test_mode)

# pad and truncate to a max_sequence_length
train_X = pad_sequences(train_X, maxlen=max_sentence_length,
                        value=0.)  # padding to max length

train_X = np.array(train_X)
train_Y = np.array(train_Y)
print("num_examples:", len(train_X), ";X.shape:",
      train_X.shape, ";Y.shape:", train_Y.shape)

# step 1: get (X,y)
test_data_x, test_data_y = get_data_x_y(base_path + 'svm/test.json')
test_X, test_Y = get_X_Y(test_data_x, test_data_y,
                         label_size, test_mode=test_mode)

# pad and truncate to a max_sequence_length
test_X = pad_sequences(test_X, maxlen=max_sentence_length,
                       value=0.)  # padding to max length

test_X = np.array(test_X)
test_Y = np.array(test_Y)
print("num_examples:", len(test_X), ";X.shape:",
      test_X.shape, ";Y.shape:", test_Y.shape)

valid_X, valid_Y = test_X, test_Y

# step 3: save to file system
save_data(cache_path_h5py, cache_path_pickle, word2index, label2index,
          train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
print("save cache files to file system successfully!")
