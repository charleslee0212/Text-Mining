#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:04:49 2020

@author: charleslee
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import random
from nltk.probability import FreqDist

import csv
import re

trainingSetN = []
testDataN = []
trainingSetP = []
testDataP = []
bagNegFreqDist = FreqDist()
bagPosFreqDist = FreqDist()
tn = 0
fn = 0
tp = 0
fp = 0

defaultStopwords = stopwords.words('english') 

def grouped(name, file, string):
    reviews  = []
    nopunct = []
    for i in range(len(file)):
        if file[i][string] == name:
            reviews.append(file[i]['Review Text'])
    
    for r in reviews:
        r.lower()
        nopunct.append(re.sub(r'[.,:;!?&-_"]', '', r))
   
    return nopunct

def count_W_inList(w,AFreqDist):
    count = AFreqDist[w]
    return count

def crossVal(data, string):
    global trainingSetN
    global testDataN
    global trainingSetP
    global testDataP
    temp = data.copy()
    
    if string == 'neg':
        for i in range(200):
            rand = random.randint(0, len(temp)-1)
            testDataN.append(temp[rand])
            del temp[rand]
        trainingSetN = temp
    else:
        for i in range(200):
            rand = random.randint(0, len(temp)-1)
            testDataP.append(temp[rand])
            del temp[rand]
        trainingSetP = temp
    

def bayes(finalProbNeg, finalProbPos, test, denominatorNeg, denominatorPos, string):
    global bagNegFreqDist
    global bagPosFreqDist
    global tn
    global fn
    global tp
    global fp
    
    likelihoodW_Neg = {} 
    likelihoodW_Pos = {} 
    
    for w in test:
        likelihoodW_Neg[w] =  (count_W_inList(w,bagNegFreqDist) + 1) / denominatorNeg
    for w in test:
        likelihoodW_Pos[w] = (count_W_inList(w,bagPosFreqDist) + 1) / denominatorPos

    for w in test:
        finalProbNeg *=  likelihoodW_Neg[w]
    print("Prob NEGATIVE = " + str(finalProbNeg) )
    
    for w in test:
        finalProbPos *= likelihoodW_Pos[w]
    print("Prob POSITIVE = " + str(finalProbPos) )
    
    if finalProbNeg > finalProbPos:
        print("\nModel predicts the test query belongs in the NEGATIVE class")
        if string == "neg":
            tn += 1
        else:
            fn += 1
    else:
        print("\nModel predicts the test query belongs in the POSITIVE class")
        if string == "pos":
            tp += 1
        else:
            fp += 1


ratings = set()
listOfDicts = []

with open("cleanReviewsFile.csv", 'r') as csvfile:
    csvreader  = csv.DictReader(csvfile, delimiter=',')
    
    for row in csvreader:
        ratings.add(row.get('Rating'))
        
        listOfDicts.append(row)
        
negative = grouped('1', listOfDicts, 'Rating')
positive = grouped('5', listOfDicts, 'Rating')

newNeg = []
for s in negative:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize(s)
    
    for w in noPunct:
        if w.lower() not in defaultStopwords:
            new += w.lower()
            new += ' '
    newNeg.append(new)

newPos = []
for s in positive:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize(s)
    
    for w in noPunct:
        if w.lower() not in defaultStopwords:
            new += w.lower()
            new += ' '
    newPos.append(new)

crossVal(newNeg, 'neg')
crossVal(newPos, 'pos')

V = set()  # the vocabulary

bagNeg= []
for s in trainingSetN:
    tokens = nltk.word_tokenize(s)
    for w in tokens:
        V.add(w)
        bagNeg.append(w)
        bagNegFreqDist[w] += 1
    
bagPos = []        
for s in trainingSetP:
    tokens = nltk.word_tokenize(s)
    for w in tokens:
        V.add(w)
        bagPos.append(w)
        bagPosFreqDist[w] += 1
        
total = len(bagNeg) + len(bagPos)
ProbNeg = len(bagNeg)/total
ProbPos  = len(bagPos)/total

 #Create denominator for Negative
denominatorNeg = 0
for w in V:
    denominatorNeg += (count_W_inList(w, bagNegFreqDist)+1)

#Create denominator for Positive
denominatorPos = 0
for w in V:
    denominatorPos += ( count_W_inList(w,bagPosFreqDist) + 1)

print("\n\n")
for t in testDataN:
    fullTestList = nltk.word_tokenize(t)
    test = [ w for w in fullTestList if w in V] 
    
    bayes(ProbNeg, ProbPos, test, denominatorNeg, denominatorPos, "neg")
    print("\n")
    
for t in testDataP:
    fullTestList = nltk.word_tokenize(t)
    test = [ w for w in fullTestList if w in V] 
    
    bayes(ProbNeg, ProbPos, test, denominatorNeg, denominatorPos, "pos")
    print("\n")
    
print("\n\n")

print("RESULTS: ")
print("tn: {}".format(tn))
print("fn: {}".format(fn))
print("tp: {}".format(tp))
print("fp: {}".format(fp))

print("\n")

precision_positive = tp/(tp+fp)
precision_negative = tn/(tn+fn)

recall_positive = tp/(tp+fn)
recall_negative = tn/(tn+fp)

accuracy = (tn+tp)/(tn+tp+fn+fp)

print("Precision Negative: {}".format(precision_negative))
print("Precision Positive: {}".format(precision_positive))
print("Recall Negative: {}".format(recall_negative))
print("Recall Positive: {}".format(recall_positive))
print("accuracy: {}".format(accuracy))
   