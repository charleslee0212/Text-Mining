#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:47:22 2020

@author: charleslee
"""

import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from nltk import word_tokenize

def convertToWNTag( inputTag ):
    # Wordnet only has 4 tags {verb,adjective,adverb,noun}, where the defualt is N for noun
    if inputTag[0] == 'N':
        returnTag = wordnet.NOUN
    if inputTag[0] == 'V':
        returnTag = wordnet.VERB
    if inputTag[0] == 'J':
        returnTag = wordnet.ADJ
    if inputTag[0] == 'R':
        returnTag = wordnet.ADV
    if inputTag[0] not in ( 'V', 'J', 'R', 'N'):
        returnTag = wordnet.NOUN
    return ( returnTag )

def convertToWordnetPOS( taggedWord ):
    word = taggedWord[0] 
    # word is the first part of the tupletag = taggedWord[1][0] 
    # first letter of the tag, i.e. first letter of secondpart of tuple
    # Wordnet only has 4 tags {verb,adjective,adverb,noun}, where the defualt is N for noun
    if tag == 'N':
        returnTag = wordnet.NOUN
    if tag == 'V':
        returnTag = wordnet.VERB
    if tag == 'J':
        returnTag = wordnet.ADJ
    if tag == 'R':
        returnTag = wordnet.ADV
    if tag not in ( 'V', 'J', 'R', 'N'):
        returnTag = wordnet.NOUN
    return ( (word,returnTag) )

def convertTaggedTokensToWordnet( taggedTokens):
    returnList = []
    for t in taggedTokens:
        wntagged = convertToWordnetPOS( t )
        returnList.append(wntagged)
    return( returnList )


sentences = [

"The woman is eating pasta.",

"The woman was eating steak.",

"The woman ate all the pasta.",

"The woman slept for days.",

"She is still sleeping.",

"She will be sleeping later.",

"She sleeps peacefully.",

"That boy sings beautifully.",

"He sang last night.",

"He has sung in school before."

]

WNlemma = nltk.WordNetLemmatizer()

setT = set()
setL = set()
setPOSL = set()

for sentence in sentences:
    T = nltk.word_tokenize(sentence)
    L = [WNlemma.lemmatize(t) for t in T]
    taggedTokens = nltk.pos_tag(T)
    POSL = [ WNlemma.lemmatize( w[0], convertToWNTag(w[1])) for w in taggedTokens] 
    
    print(sentence)
    print(T)
    print(L)
    print(POSL)
    
    for t in T:
        setT.add(t)
    for l in L:
        setL.add(l)
    for p in POSL:
        setPOSL.add(p)
    
    print("\n")
    
print("setT: ")
print(setT)
print(len(setT))
print("\n")

print("setL: ")
print(setL)
print(len(setL))
print("\n")

print("setPOSL")
print(setPOSL)
print(len(setPOSL))
print("\n")
    
print("setT - setL: ")
print(setT.difference(setL))
print("\n")

print("setL - setPOSL: ")
print(setL.difference(setPOSL))

