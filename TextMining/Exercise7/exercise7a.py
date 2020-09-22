'''
Exercise 7a

This code is a subset of the naive bayes program. The goal of this exercise is to understand the computational complexity
issues that arise from "sloppy" programming when dealing with larger data sets.  This code will run a subset of the model for
one test query, yet for a training set of 12000 small documents (women's clothes review text) it take four minutes to run 
and a one-year old high end macbook pro.

The goal here is to understand what the sloppy mistake is and fix it.

We will discuss this in class (if you miss class watch the recording of the 5/13/2000 class before proceeding).
Your job is to code up a much faster version with minimal changes to the code!

'''

import nltk
from nltk.probability import FreqDist
import time

start_time = time.time()

# def count_W_inList(w, aList):
# 	count = 0
# 	for w2 in aList:
# 		if w2 == w:
# 			count += 1
# 	return(count)

fullTestString = 'I love these pants!  The material is so comfy yet the look super.'
test = nltk.word_tokenize(fullTestString)

allPositive = []   # to hold all the positive training documents, a list of strings

# read in the positive training documents
fp = open('posTrain12000.txt', 'r') 
for line in fp:
	allPositive.append(line)
	
V = set()  # the vocabulary
bagPositive = []   # to hold the bag of words for all the training documents

for s in allPositive:
	tokens = nltk.word_tokenize(s)
	for w in tokens:
		V.add(w)   # V also has the negative training words added to it, not relevant to to this exercise
		bagPositive.append(w)

fdist = FreqDist()
for w in bagPositive:
    fdist[w] += 1

print("len(allPositive), which is the number of lines in the postive training set = " + str(len(allPositive)) )
print("len(bagPositive) = " + str(len(bagPositive)) )
print("len(V) = " + str(len(V)))


# Now a subset of the naive bayes model calculations:

# create the denominator for the positive class
# for each word in the complete training vocabulary, sum up (count(w,c) + 1)
denominator = 0
for w in V:
	denominator += (fdist[w] + 1)

# for each TEST document word w, get the likelihood[w,postive]
likelihoodW_pos = {} # dictionary of probabilities for positive class
for w in test:
	likelihoodW_pos[w] =  (fdist[w] + 1) / denominator

# denominator and likelihood(w) for negative skipped because not relevant to this exercise
# final calculations of probabilities skipped because not relevant to this exercise....
print(str((time.time() - start_time)) + " seconds")
print("DONE")
