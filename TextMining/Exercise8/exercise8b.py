#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:26:51 2020

@author: charleslee
"""


import exercise8 as functions
import nltk
import math

tests = ['the movie was horrible,  I hated it!',
        'It was a movie',
        'it had a wonderful plot',
        'the movie was wonderful, I loved it!'
        ]

Y = [0,0,1,1]

positiveWordSet = set()
negativeWordSet = set()

with open("hu_liu_positiveLexicon.txt",mode='r',encoding="ISO-8859-1") as fp:
    word = fp.readline()
    while word:
        positiveWordSet.add(nltk.word_tokenize(word)[0])
        word = fp.readline()
        
with open("hu_liu_negativeLexicon.txt",mode='r',encoding="ISO-8859-1") as fp:
    word = fp.readline()
    while word:
        negativeWordSet.add(nltk.word_tokenize(word)[0])
        word = fp.readline()

def create_X(documents, pos, neg):
    X = []
    for doc in documents:
        fv = functions.generate_fv(doc, pos, neg)
        X.append(fv)
    return X

def compute_gradient(fv, w, b, y):
    diff = functions.prob_y_1(fv, w, b) - y
    gradient = []
    for f in fv:
        gradient.append(f*diff)
    return gradient

def LCE(y_prediction, y):
    lce = -1 * (y * math.log(y_prediction) + (1 - y) * math.log(1 - y_prediction))
    return lce

def gradient_descent_stochastics(X, Y, learning_rate, itr):
    w = [0, 0, 0]
    b = 0
    for i in range(itr):
        loss = 0
        for j in range(len(X)):
            y_prediction = functions.prob_y_1(X[j], w, b)
            loss += LCE(y_prediction, Y[j])
            gradient = compute_gradient(X[j], w, b, Y[j])
            changes_w = [grad * learning_rate for grad in gradient]
            b -= (learning_rate) * (y_prediction - Y[j])
            for k in range(len(w)):
                w[k] -= changes_w[k]
        if i < 10:
            print("cost at i: {} is {}".format(i, (1/len(X)) * loss))
            print("w: {} b: {}".format(w, b))
        elif i % 20 == 10:
            print("cost at i: {} is {}".format(i, (1/len(X)) * loss))
            print("w: {} b: {}".format(w, b))
    return (w, b)

w, b = gradient_descent_stochastics(create_X(tests, positiveWordSet, negativeWordSet), Y, 0.01, 400)
print("Optimal W: [] Optimal b: {}".format(w, b))
for s in tests:
    functions.classify_doc(s, positiveWordSet, negativeWordSet, w, b)