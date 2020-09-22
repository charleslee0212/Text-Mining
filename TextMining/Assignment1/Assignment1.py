#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:03:21 2020

@author: charleslee
"""

import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

rcount = 0

bad_words = set()
good_words = set()

temp_words = open("bad_words.txt", "r")

for word in temp_words:
    word = re.sub(r'\n', '', word)
    bad_words.add(word)

temp_words = open("good_words.txt", "r")

for word in temp_words:
    word = re.sub(r'\n', '', word)
    good_words.add(word)


# bad_words = {'bad', 'disappointed', 'disappointment', 'however', 'but', 'not', 
#              'cheap', 'unfortunately', 'unflattering', 'returning', 'problem', 
#              'weird', 'nothing', 'disappointing', 'ill', 'wrong', 'sad','least', 
#              'issue', 'shapeless', 'itchy', 'never', 'oversized', 'flimsy', 'complaint',
#              'poorly', 'disappointment', 'poor', 'awful', 'terrible', 'sadly', 'otherwise'
#              'old', 'worst', 'horrible', 'awkward', 'unwearable', 'shame', 'waste',
#              'ridiculous', 'mess', 'terribly', 'wrinkled', 'ridiculously'}

# good_words = {'love', 'great', 'like', 'perfect', 'flattering', 'soft', 'comfortable',
#               'beautiful', 'nice', 'cute', 'super', 'perfectly', 'pretty', 'compliments'
#               'gorgeous', 'comfy', 'better', 'lovely', 'absolutely', 'happy', 'easy',
#               'loved', 'glad', 'nicely', 'unique', 'warm', 'fitted', 'favorite', 'highly'
#               'amazing', 'beautifully', 'adorable', 'cool', 'exactly', 'hot', 'wonderful'
#               'stunning', 'special', 'elegant', 'pleased', 'sexy', 'fabulous', 'classy'
#               'justice', 'good', 'vibrant'}

def grouped(name, file, string):
    global rcount
    rcount = 0
    reviews  = ""
    for i in range(len(file)):
        if file[i][string] == name:
            rcount += 1
            reviews += (file[i]['Review Text'])
            
    reviews.lower()
    nopunct = re.sub(r'[.,:;!?&-_]', '', reviews)
   
    return nopunct

def mostCW(words, num):
    words = re.split('\s', words)
    defaultStopwords = stopwords.words('english')
    fdist = FreqDist()
    for word in words:
        if word not in defaultStopwords:
            fdist[word] += 1
    
    if '' in fdist:
        fdist.pop('')
    
    return fdist.most_common(num)

def avgLen(words):
    words = re.split('\s', words)
    return len(words) / rcount

def avgWords(words, compareWords):
    words = re.split('\s', words)
    count = 0;
    for word in words:
        if word in compareWords:
            count += 1
    return count / len(words)
    
    

classNames = set()
ratings = set()

with open("Womens Clothing E-Commerce Reviews.csv", 'r') as csvfile:

    csvreader  = csv.DictReader(csvfile, delimiter=',')  # create a "DictReader" object

 

   # print(csvreader.fieldnames)

 

    listOfDicts = []  # another way would be to use panda data frames,

    #print("\n\n")

 

    # lets add each row of the file to "listOfDicts"

    count = 0

    for row in csvreader:
        
        classNames.add(row.get('Class Name'))
        ratings.add(row.get('Rating'))
        
        count += 1

        # if len(row) != 11:

        #     print("ERROR - should have eleven felds!, len(row) = " + str(len(row)))

        # if ((count % 1000) == 0):

        #     print("\n ---------- " + str(count))

        #     print(csvreader.fieldnames)

        #     print("class = " + str(row['Class Name']) )

        #     print(row)

        listOfDicts.append(row)

    print("At end, len of listOfDicts = ")

    print( len(listOfDicts)) 
    
    print("\n\n")
    
    print("Classname: \n")
    for name in classNames:
        
        if name == '':
            print('Class Name Unknown')
        else:
            print(name)
        
        words = grouped(name, listOfDicts, 'Class Name')
        print("Most Common word: " + str(mostCW(words, 1)))
        print("Average length of review: " + str(avgLen(words)))
        print("Average number of good words: " + str(avgWords(words, good_words)))
        print("Average number of bad words: " + str(avgWords(words, bad_words)))
        print("\n")
        
    print("Rating: \n")
    for num in ratings:
        print(num)
        words = grouped(num, listOfDicts, 'Rating')
        print("Most Common word: " + str(mostCW(words, 1)))
        print("Average length of review: " + str(avgLen(words)))
        print("Average number of good words: " + str(avgWords(words, good_words)))
        print("Average number of bad words: " + str(avgWords(words, bad_words)))
        print("\n")
    


    
    
    
    
    

    
        

        
        
        