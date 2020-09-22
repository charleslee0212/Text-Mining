# Assignment2_Part2_v5.py

# This version is divorced from the previous version reliance on the clothesReview .csv files for input.
# Instead, it just assume clear text input files - one document (a single string with now \n chars) per line.
# It assume four input files exist:   
#	posTrain.text	- the postive training set
#	negTrain.text	- the negative training set
#	posTest.text	- the positive test set
#	negTest.text	- the negative test set



# This like _v4, this version also differs from version _v3 in that the bag of words has been replaced with a FreqDist. (see below)
# This change GREATLY reduces size and hence run time.
# In _v3, each word of a query is counted in the bag of words using count_W_inList()
# In _v4, each word of a query is counted in the bag of words using count_W_inFreqDist()
# To give an example of the savings, in one experiment I ran the len(bagOfWords) ~= 35,000 while the len(FreqDist) ~= 3700
# Further, the FreqDist uses a hashmap to store/retrieve items, so it is O(1) to do a lookup versus 0(N).
#
# When run the old way (using a list for the bag of words), maximum runtime image was 101MB and the runtime was 21:21.69 with 100% processor usage
# When run the new way (using a FreqDist for bag of words), the runtime image was 7MB and the runtime was 00:02.82 with 92% processor usage
# In otherwords 1281 seconds versus 3 seconds, or 400x faster!


'''
Questions to explore:

- Is there a difference in accuracy if I do/not remove stop words?
- Is there a difference in accuracy if I do/not remove puncuation?
- How about if I use bigrams of the reviews instead?  Improved accuracy?
- Is there a difference based on class name (i.e. dresses versus blouses versus jeans)
- If I compare rating 1 verus rating 5 does that yield more accuracy thatn 2 verus 4?

'''

# This code use AllPositive to be == "postive" 
# 	and AllNegative to be == "negative" 
# You will see both "postive" and "AllPositive" used, likewise with negative

import nltk
from nltk import FreqDist
import math
import csv
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.corpus import stopwords
defaultStopwords = stopwords.words('english')   # can get other languages
from nltk.tokenize import RegexpTokenizer
from random import randint


'''
# OLD version = assumng a list
# helper function to count how many times a word is in a list
def count_W_inList(w, aList):
	count = 0
	for w2 in aList:
		if w2 == w:
			count += 1
	return(count)
'''

# helper function to count how many times a word is in a list
# new version - freqDist
def count_W_inFreqDist(w, aDist):
	return(aDist[w])


# global vars to hold training and test sets
AllPositive = []
AllNegative = []

# global vars used in initModel and runModel
V = set()
VPositive = set() # vocab positive set
VNegative = set() # vocab negative set
bagPositiveFreqDist = FreqDist()
bagNegativeFreqDist = FreqDist()


#-------------------------
def initModel(verbose = True):

	if (verbose):
		print("Starting initModel(), verbose = " + str(verbose))
	
	# V is the vocabulary, i.e. set { } of all words (duplicates removed because it is a set)
	# C is the set of classes:   C = { CP, CN }, where 
	#    CP = class postive, i.e. the class of positve reviews 
	#    CN = class negative, i.e. the class of negative reviews 
	
	global V 
	global VPositive  # vocabulary postive documents as a set
	global VNegative 
	global bagPositiveFreqDist
	global bagNegativeFreqDist
	global AllPositive  # all of the positive document strings  as an list
	global AllNegative
	
	
	for s in AllNegative:
		tokens = nltk.word_tokenize(s)
		for w in tokens:
			bagNegativeFreqDist[w] += 1
			V.add(w)
			VNegative.add(w)
	
	for s in AllPositive:
		tokens = nltk.word_tokenize(s)
		for w in tokens:
			bagPositiveFreqDist[w] += 1
			V.add(w)
			VPositive.add(w)
	
	if (verbose):


		print("len(VPositive) = " + str( len(VPositive)))
		# print("\nVPositive = ")
		# print(VPositive)


		print("len(VNegative) = " + str( len(VNegative)))
		# print("\nVNegative = ")
		# print(VNegative)

		print("len(V) = " + str( len(V)))
		#print("\nV = ") 
		# print(V)

		print("\n\nbagNegativeFreqDist")
		print(bagNegativeFreqDist)
		print(bagNegativeFreqDist.most_common(100))
		print("\n\nbagPositiveFreqDist")
		print(bagPositiveFreqDist)
		print(bagPositiveFreqDist.most_common(100))

		print("\n\n bagPositiveFreqDist['love'] ")
		print( bagPositiveFreqDist['love'] )
	


#-------------------------
def runModel(verbose = True):

	ProbPositive = len(AllPositive) / (len(AllPositive) + len(AllNegative) )
	ProbNegative  = len(AllNegative) / (len(AllPositive) + len(AllNegative) )

	fullTestList = nltk.word_tokenize(fullTestString)
	test = [ w for w in fullTestList if w in V]     # ZZZ ??? - Probably need to remove duplicates (set)
	if (verbose):
		print("\nTest query = ")
		print(test)
	
	if (len(test) == 0):
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("ZZZZZZZZZZZZ -> test empty!!!!!!!!")
		print("fullTestList = " + fullTestString)


	likelihoodW_pos = {} # dictionary of probabilities for positive class
	
	# create the denominator for the positive class
	# for each word in the complete vocabulary, sum up (count(w,c) + 1)
	denominator = 0
	for w in V:
		denominator += ( count_W_inFreqDist(w,bagPositiveFreqDist) + 1)
	if (verbose):
		print("verbose: denominator postive class = " + str(denominator))
	
	# for each query word w, get the likelihood[w,postive]
	for w in test:
		likelihoodW_pos[w] =  (count_W_inFreqDist(w,bagPositiveFreqDist) + 1) / denominator
		if (verbose):
			print("verbose: numerator postive class for w = " +  w + "  = " + str(count_W_inFreqDist(w,bagPositiveFreqDist) + 1))
	
	if (verbose):
		print("likelihoodW_pos = ")
		print(likelihoodW_pos)
	
	likelihoodW_neg = {} # dictionary of probabilities for negative class
	
	# create the denominator for the negative class
	# for each word in the complete vocabulary, sum up (count(w,c) + 1)
	denominator = 0
	for w in V:
		denominator += ( count_W_inFreqDist(w,bagNegativeFreqDist) + 1)
	if (verbose):
		print("verbose: denominator negative class = " + str(denominator))
	
	# for each query word w, get the likelihood[w,negative]
	for w in test:
		likelihoodW_neg[w] =  (count_W_inFreqDist(w,bagNegativeFreqDist) + 1) / denominator
		if (verbose):
			print("verbose: numerator negative class for w = " +  w + "  = " + str(count_W_inFreqDist(w,bagNegativeFreqDist) + 1))
	
	if (verbose):
		print("likelihoodW_neg = ")
		print(likelihoodW_neg)
	
	
	# final calculations
	
	# P(-) P(S | -) 
	# where S is the test query Sentence, S = "predictable with no fun"
	finalProbNegative = math.log(ProbNegative,10)
	for w in test:
		finalProbNegative +=  math.log(likelihoodW_neg[w],10)
	if (verbose):
		print("Prob function for negative class using logs = " + str(finalProbNegative) )
	
	# P(+) P(S | +) 
	# where S is the test query Sentence, S = "predictable with no fun"
	finalProbPositive = math.log(ProbPositive,10)
	for w in test:
		finalProbPositive +=  math.log( likelihoodW_pos[w], 10)
	if (verbose):
		print("Prob function for positive class using logs = " + str(finalProbPositive) )
	
	if (verbose):
		print("VPositive - VNegative = ")
		print(VPositive.difference(VNegative))
		print("VNegative - VPositive = ")
		print(VNegative.difference(VPositive))
		print("len(AllPositive) = " + str( len(AllPositive)))
		print("len(AllNegative) = " + str( len(AllNegative)))

	if (finalProbPositive > finalProbNegative):
		print("Leaving runModel(): Model predicts the test query belongs in the Positive class")
		return('positive')
	else:
		print("Leaving runModel(): Model predicts the test query belongs in the Negative class")
		return('negative')
	
	
# ------ end of "runModel() ---------------------


# ------------- begin of initialize()
def initialize(verbose='False'):
# creates and fills for global lists: 
#        AllPositive, globalTest


	positiveDocs = []
	negativeDocs = []
	
	# read in, tokenize, and remove stop words for posTrain.txt => file of postive documents
	fpPosTrain = open('posTrain.txt', 'r') 
	for line in fpPosTrain:
		positiveDocs.append(line)
	# Now remove all stopwords
	newPositive = []
	for s in positiveDocs:
		new = ''
		tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
		noPunct = tokenizer.tokenize( s )
		for word in noPunct:
			if word.lower() not in defaultStopwords:    
				new += word.lower()
				new += ' '
		newPositive.append(new)
	if verbose:
		print("\n\nnewPositive = ")
		print(newPositive)
		print("\n\n")
	
	# read in, tokenize, and remove stop words for negTrain.txt => file of negative documents 
	fpNegTrain = open('negTrain.txt', 'r') 
	for row in fpNegTrain:
		negativeDocs.append(row)
	# Now remove all stopwords
	newNegative = []
	for s in negativeDocs:
		new = ''
		tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
		noPunct = tokenizer.tokenize( s )
		for word in noPunct:
			if word.lower() not in defaultStopwords:    
				new += word.lower()
				new += ' '
		newNegative.append(new)
	if verbose:
		print("\n\nnewNegative = ")
		print(newNegative)
		print("\n\n")
	
	# read in, tokenize, and remove stop words for negTest.txt => file of negative TEST documents 
	tempNegativeTestDocs = []
	fpNegTest = open('negTest.txt', 'r') 
	for row in fpNegTest:
		tempNegativeTestDocs.append(row)
	# Now remove all stopwords
	newNegativeTestDocs = []
	for s in tempNegativeTestDocs:
		new = ''
		tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
		noPunct = tokenizer.tokenize( s )
		for word in noPunct:
			if word.lower() not in defaultStopwords:    
				new += word.lower()
				new += ' '
		newNegativeTestDocs.append(new)
	
	# read in, tokenize, and remove stop words for posTest.txt => file of negative TEST documents 
	tempPositiveTestDocs = []
	fpPosTest = open('posTest.txt', 'r') 
	for row in fpPosTest:
		tempPositiveTestDocs.append(row)
	# Now remove all stopwords
	newPositiveTestDocs = []
	for s in tempPositiveTestDocs:
		new = ''
		tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
		noPunct = tokenizer.tokenize( s )
		for word in noPunct:
			if word.lower() not in defaultStopwords:    
				new += word.lower()
				new += ' '
		newPositiveTestDocs.append(new)
	

	# deep copy
	global AllPositive 
	AllPositive = [ s for s in newPositive]  
	
	# example to reduce size when posTrain.txt is very large - hence slow!
	# Instead, make 1/10the size
	# AllPositive = []
	# for i in range(0, len(newPositive), 10):
		# AllPositive.append(newPositive[i]) 
	print("len(AllPositive) = " + str(len(AllPositive)))
	
	# deep copy
	global AllNegative 
	AllNegative = [ s for s in newNegative]
	print("len(AllNegative) = " + str(len(AllNegative)))
	
	# deep copy
	global testDocsNegative 
	testDocsNegative = [ s for s in newNegativeTestDocs]
	print("len(testDocsNegative) = " + str(len(testDocsNegative)))
	
	# deep copy
	global testDocsPositive 
	testDocsPositive = [ s for s in newPositiveTestDocs]
	print("len(testDocsPositive) = " + str(len(testDocsPositive)))
	
	
	if verbose:
		print("About to leave initialize():")
	
	return()
# ------------- end of initialize()
	

	
	

# START OF MAIN

debug = False

# call initital to create AllPositive, AllNegative 
initialize(False)
if (debug):
	print("\n\nAfter calling initialize():")
	print("len(AllNegative) = " + str(len(AllNegative)) )
	# print("AllNegative = ")
	# print(AllNegative)
	print("len(AllPositive) = " + str(len(AllPositive)) )
	# print("AllPositive = ")
	# print(AllPositive)


initModel(True)  # create the bags of words, global vars, so do not redo every time run model



'''
# simple tests
fullTestString = 'Really cute, I love it.'
fullTestString = 'Did not fit, return it, ugly'
runModel(False)
'''


truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

print("\n\nlen(testDocsPositive) = " + str(len(testDocsPositive)))
print("Now running Positive test documents:")
for s in testDocsPositive:
	fullTestString = s
	print("\nTest document = " + s)
	returnString = runModel(False)
	if (returnString == 'positive'):
		truePositive += 1
	else:
		falseNegative += 1

print("\n\nNow running Negative test documents:")
for s in testDocsNegative:
	fullTestString = s
	print("\nTest document = " + s)
	returnString = runModel(False)
	if (returnString == 'negative'):
		trueNegative += 1
	else:
		falsePositive += 1


print("\n\n\n")
print("len(trainDocsPositive) = " + str(len(AllPositive)) )
print("len(testDocsNegative) = " + str(len(testDocsNegative)) )
print("len(trainDocsNegative) = " + str(len(AllNegative)) )
print("len(testDocsPositive) = " + str(len(testDocsPositive)) )
print("\n")
print("truePositive = " + str(truePositive))
print("falseNegative = " + str(falseNegative))
print("trueNegative = " + str(trueNegative))
print("falsePositive = " + str(falsePositive))

print("pos precision = " + str(  (truePositive ) /  (truePositive+falsePositive) ))
print("pos recall = " + str(  (truePositive ) /  (truePositive+falseNegative) ))

print("neg precision = " + str(  (trueNegative ) /  (trueNegative+falseNegative) ))
print("neg recall = " + str(  (trueNegative ) /  (trueNegative+falsePositive) ))

print("accuracy = " + str(  (truePositive + trueNegative) /  (truePositive+trueNegative+falsePositive+falseNegative) ))

print(len(AllPositive))
print(len(AllNegative))
