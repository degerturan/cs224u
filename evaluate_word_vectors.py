#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import scipy
from scipy.spatial import distance
import csv
import numpy as np 
import matplotlib.pyplot as plt
import random
import math
from math import exp
import copy
from collections import Counter
from itertools import *
import itertools
import scipy.io.wavfile as wavfile
import argparse
from sklearn.manifold import TSNE
import morph_parse_german
from sklearn import cross_validation
from sklearn import svm

SPLIT_MORPH = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='glove/vocab_german_all.txt', type=str)
    parser.add_argument('--vectors_file', default='glove/vectors_german_all.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = map(float, vals[1:])

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.iteritems():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, vocab, ivocab, args.vocab_file)


def evaluate_vectors(W, vocab, ivocab, filename):
    """Mostly visualization"""

    def plot_embeddings(words, title):
        vs = []
        for w in words:
            if isinstance(w, tuple):
                vs.append(np.mean(np.array([W[vocab[x]] for x in w]) , axis=0))
            else:
                vs.append(W[vocab[w]])

        vs = np.array(vs) #([W[vocab[w]] for w in words])

        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        model.fit_transform(vs)

        plt.figure()
        plt.scatter(vs[:,0],vs[:,1])

        for i, w in enumerate(words):
            if isinstance(w, tuple):
                w = "+".join(w)
            plt.annotate(w, xy=(vs[i,0],vs[i,1]))

        plt.savefig("tsne-"+title+filename.split('.')[0].split('/')[1]+".png", format = 'png')
        plt.close()

    plot_embeddings(["raum", "fahrt", "weltall", "zivilisation"], "1")
    plot_embeddings(["klass", "zimm", "lesung", "saal", "vor", "raum"], "2")
    plot_embeddings([("raum", "fahrt"), "weltall", ("klass", "raum"), ("vor", "lesung", "saal")], "3")
    plot_embeddings([("raum", "fahrt", "zivilisation"), "weltall", ("klass", "raum"), ("vor", "lesung", "saal")], "4")
    plot_embeddings([("raum", "fahrt", "zivilisation"), "weltall", ("klass", "raum"), ("vor", "lesung", "saal"), "schul", "grund","schul", "universitat"], "5")
    # plot_embeddings([("raum", "fahrt", "zivilisation"), "weltall", ("klass", "raum"), ("vor", "lesung", "saal"), "schul", "grundschul", "universitat"], "5")
    plot_embeddings(["autobahn", ("auto", "bahn"), "auto", "bahn"], "5")
    plot_embeddings(["produkt", "ion", "landschaft", "fabrik", "manufaktur", ("produkt", "ion", "landschaft"), ("produkt", "ion"), ("produkt", "landschaft")], "6")
    plot_embeddings(['organisation', 'struktur', ('organisation', 'struktur'), "firma", ("firma", "struktur"), "unter","nehm", ("unter","nehm", "struktur")], "7")
    # plot_embeddings(['organisation', 'struktur', ('organisation', 'struktur'), "firma", ("firma", "struktur"), "unternehmen", ("unternehmen", "struktur")], "7_unprocessed")

    # organisationstruktur was pruned!
    # "raum"+"fahrt"
    # weltall
    # "klass"+"raum"
    # vor+lesung+saal
    # class + room
    # lecture + hall
    # Wissenschaftler
    # Sozialwissenschaft
    # ... wissenschaft
    # Regierungseinrichtungen
    # Aufklärungskampagne
    # Organisationstruktur
    # Kommunikationtechnologie
    # Raum<#>fahrt<#>zivilisation

    def evaluate_sentiment():
        N = 500 # num of examples (approx)
        n = 50 # feature dimension

        # 0 to 5
        pos_labels = np.loadtxt('data/GermanSentimentData/positive-labels.txt', skiprows=1)

        X = []# np.empty((N,n)) # data
        Y = [] #np.empty(N, dtype=bool) # labels

        with open('data/GermanSentimentData/data.txt', 'r') as f:
            for i, line in enumerate(f):
                sys.stdout.write("  Building training data: %d of %d sentences.   \r" % (i+1, N))
                sys.stdout.flush()

                # morph split words in this sentence

                if SPLIT_MORPH:
                    words = morph_parse_german.split_sentence(line)
                else:
                    words = [morph_parse_german.stem(w) for w in line.split(' ')]

                vals = [W[vocab[x]] for x in words if x in vocab]
                if (len(vals) == 0):
                    continue
                else:
                    X.append(np.mean(np.array(vals), axis=0))

                # Compute label
                # TODO: consider negative labels as well
                pos = pos_labels[i]
                # TODO: probably ignore examples with big discrepancies between labelers
                score = np.mean(pos)
                Y.append(bool(score > 2.5))

        N = len(Y)
        Y = np.array(Y)
        X = np.array(X)
        assert X.shape == (N,n)

        # TODO: train model & test model!
        clf = svm.SVC(kernel='linear', C=1)
        # cv = cross_validation.ShuffleSplit(N, n_iter=3,test_size=0.3, random_state=0)
        scores = cross_validation.cross_val_score(clf, X, Y, cv=10)

        print scores
        print "Accuracy: "+str(scores.mean())+" (+/- "+str(scores.std())+")"

    evaluate_sentiment()

if __name__ == "__main__":
    main()
