#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:15:40 2020

@author: charleslee
"""

import nltk
import math

b = 0

w = [1, -2, -0.01]

def zValue(fv, w, b):
    z = 0
    for i in range(len(fv)):
        z += (fv[i] * w[i])
        
    z += b
    return z

def sigmoid(N):
    y = 1/(1 + math.exp(N * -1))
    #print("y: " + str(y))
    return y

def prob_y_1(fv, w, b):
    z = zValue(fv, w, b)
    #print("z: " + str(z))
    return sigmoid(z)

def generate_fv(s, pos, neg):
    fv = []
    tokens = nltk.word_tokenize(s)
    print("Tokens: " + str(tokens))
    countPos = 0
    countNeg = 0
    for w in tokens:
        if w.lower() in pos:
            countPos += 1
        elif w.lower() in neg:
            countNeg += 1
    
    fv.append(countPos)
    fv.append(countNeg)
    fv.append(len(s))
    print("Feature Vector: " + str(fv))
    return fv

def classify(fv, w, b):
    prob = prob_y_1(fv, w, b)
    if prob > 0.5:
        print("positive")
        return "positive"
    else:
        print("negative")
        return "negative"
    
def classify_doc(s, pos, neg, w, b):
    fv = generate_fv(s, pos, neg)
    return classify(fv, w, b)

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
        
tests = ['the movie was horrible,  I hated it!',
        'It was a movie',
        'it had a wonderful plot',
        'the movie was wonderful, I loved it!'
        ]

print("Length of Positive: " + str(len(positiveWordSet)))
print("Length of Negative: " + str(len(negativeWordSet)))
print("\n\n")
for s in tests:
    print("Test: " + s)
    print(classify_doc(s, positiveWordSet, negativeWordSet, w, b))
    print("\n\n")
    
    