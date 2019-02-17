#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a simple Question Classifier using TF-IDF or Bag of Words Model
"""

import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.externals import joblib

import jieba
import jieba.posseg as pseg
import gensim
import numpy

import json
from collections import defaultdict
import io
import os

from constant import id2category


def _tokenize(text):
    return list(text)
    # return jieba.lcut(text, cut_all=False)


def save_model(grid, filename='../model/svm.pkl'):
    joblib.dump(grid, filename, compress=1)


def load_model(filename='../model/svm.pkl'):
    grid = joblib.load(filename)
    return grid


def train():
    # the training data folder must be passed as first argument
    dataset_train = load_files('../data/svm/train', shuffle=False)
    dataset_test = load_files('../data/svm/test', shuffle=False)
    print("n_samples: %d" % len(dataset_train.data))

    docs_train, docs_test = dataset_train.data, dataset_test.data
    y_train, y_test = dataset_train.target, dataset_test.target

    # split the dataset in training and test set:

    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=_tokenize, ngram_range=(1, 3))),
                         ('tfidf', TfidfTransformer(use_idf=False, norm='l1')),
                         ('clf', SVC(C=100, gamma=.1, probability=True))
                         #('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
                         ])

    text_clf.fit(docs_train, y_train)

    y_predicted = text_clf.predict(docs_test)

    confusion = {}
    for i, q in enumerate(docs_test):
        if y_predicted[i] != y_test[i]:
            try:
                confusion[q.decode('utf8')] = {
                    'label': id2category[y_test[i]], 'prediction': id2category[y_predicted[i]]}
            except:
                continue
    with io.open('confusion.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(confusion, ensure_ascii=False, indent=4))

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset_test.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    save_model(text_clf)


def predict(clf, question):
    return clf.predict([question])[0]


def predict_proba(clf, question):
    return clf.predict_proba([question])


if __name__ == "__main__":
    train()
    grid = load_model()
    question = u'字符串的用法'
    print(predict(grid, question))
    print(predict_proba(grid, question))
