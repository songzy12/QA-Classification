#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a simple Question Classifier using TF-IDF or Bag of Words Model
"""

import sys
import json
import io
import os

import numpy

from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

from matplotlib import pyplot

from lightgbm import LGBMClassifier

import jieba
import jieba.posseg as pseg
import gensim

from constant import id2category


def tokenize(text):
    # return jieba.lcut(text)
    return list(text)


def save_model(grid, filename='../model/xgb.pkl'):
    joblib.dump(grid, filename, compress=1)


def load_model(filename='../model/xgb.pkl'):
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

    model = LGBMClassifier()

    print(model)
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 3))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', model)
                         ])

    text_clf.fit(docs_train, y_train)

    y_predicted = text_clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset_test.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    save_model(text_clf)

    print(model.feature_importances_)

    # plot_importance(model)
    # pyplot.show()


def predict(clf, question):
    return clf.predict([question])[0]


def predict_proba(clf, question):
    return clf.predict_proba([question])


if __name__ == "__main__":
    train()
    grid = load_model()
    question = u'数据结构当中的图怎么都弄不懂怎么办？'
    print(predict(grid, question))
    print(predict_proba(grid, question))
