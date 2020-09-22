#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:30:04 2020

@author: charleslee
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import random
from nltk.probability import FreqDist

AboutEquity = ["I would agree that poorly designed training sets can be a problems. They introduce unintended bias that affects the outcomes of the algorithms. If the algorithm is only specialized in a certain type of data, it will fail to accurate work with new types of data introduced.",

    "I agree because if we train these models to be racist or something like that then over time, they will always be racist or biased in some sort of manor. As we start to rely on these machines more and more for things, this can pose huge issues and flaws in our logic.",

    "Yes; the better you are able to represent the population your data draws from, the better prepared your program will be to accurately assess what it comes across. This makes common sense, and yet it it so easy to ignore if we are thinking outside of our normal CS bubble. Many of us don't think about or want to think about social impacts our programs can make, especially at the early stages of a project.",

    "Yes because poorly designed training sets can lead to bias where it can output inaccurate data. This can misrepresent what the program initially design for. ",

    "Yes! Person who collected the training set probably has some inherent subconscious bias and didn't think to include enough data that correctly represent the diverse range of possible data classes! ",

    "Yes they limit the potential of other people and exclude others from equal opportunities. Plus they are biased in nature, by whoever the creators are.",

    "I absolutely agree with the speaker that poorly designed training sets can be a problem, because they can perpetuate the systems of discrimination in our society. There have been literacy tests, the Grandfather Clause, gerrymandering, and voter ID laws. Society should be moving away from discriminatory practices, not inadvertently or consciously continuing them. ",

    "absolutely; such training sets spread the implicit bias that was created within the sets. Many people also put a significant amount of trust into artificial intelligence/other algorithms and if those are unreliable then people relying on them is incredibly dangerous",

    "Yes- especially with her example with the facial recognization software not being able to recognize her face due to her color is definitely an issue, especially when the software is meant for all users of all 'spectrum'. Poorly designed training sets mean that the designers are not considering all angles for their software, limiting its potential",

    "I definitely agree. Not only can poor training sets lead to bias, but can give noisy and uninteresting results",

    "Yes! Clearly poor training sets can reinforce inequity.",

    "Yes! I am terrified to see ML/AI may actually perpetuate and strengthen systemic racism instead of helping to dismantle racism.",

    "I agree. Not only poorly designed training sets create a technical problem, but the societal implications need to be considered.",

    "I agree. If training poorly designed sets result in algorithmic bias than clearly there is a social justice issue here!",

    "Yes, I agree. Clearly one can build bias into an algorithm.  We need to work towards removing bias, not codifying it in future software!"

    ]

AboutTechnology = [

    "I agree, if the training set is to narrow then the probability space will be too small resulting in a narrow model that fails to have good precision.",

    "I agree in this situation because training sets are intended to give the people using code a somewhat rounded experience of software so they can expand what it can do. If training sets are poorly designed, then it makes testing software more difficult, and then it can take away from time spent actually working with the software if you have to teach it a new set of information or to use new data. ",

    "Poorly designed training sets can definitely be a huge problem. If a training set is poorly designed, then it will not properly train people and create a chain affect of problems, which can create life threatening mistakes.",

    "I agree because a poorly designed training set will not adequately capture the all the types of queries that will be run and hence the model will fail to be a good classifier.",

    "I agree. If a data set is not sufficiently varied then the model will fail to make accurate predictions.",

    "I agree, the training set needs to have sufficient coverage and size to enable high quality classification.",

    "I agree, a training set needs to cover a sufficiently large space of words to allow for meaninful classification.",

    "I agree.  More and more machine learning is going to be used in the future, so it is important that students understand the importance of crafting robust training sets.",

    "I agree because if the sets don't encompass a wide variety, then there will be subsets that are left out when its time for the software to actually run.",

    "I agree. Poorly designed training sets could result in statisitical correlations that result in poor classification."

    ]


trainingSetE = []
testDataE = []
trainingSetT = []
testDataT = []
bagEquityFreqDist = FreqDist()
bagTechnologyFreqDist = FreqDist()
teq = 0
feq = 0
ttech = 0
ftech = 0

avgAccuracy = 0
avgRecall_eq = 0
avgRecall_tech = 0
avgPrecision_eq = 0
avgPrecision_tech = 0

defaultStopwords = stopwords.words('english')   # can get other languages

def count_W_inList(w,AFreqDist):
    count = AFreqDist[w]
    return count

def crossVal(data):
    global trainingSetE
    global testDataE
    global trainingSetT
    global testDataT
    temp = data.copy()
    if len(data) == 15:
        for i in range(12):
            rand = random.randint(0, len(temp)-1)
            trainingSetE.append(temp[rand])
            del temp[rand]
        testDataE = temp
    else:
        for i in range(8):
            rand = random.randint(0, len(temp)-1)
            trainingSetT.append(temp[rand])
            del temp[rand]
        testDataT = temp

def bayes(finalProbEq, finalProbTech, test, denominatorEq, denominatorTech, string):
    global bagEquityFreqDist
    global bagTechnologyFreqDist
    global teq
    global feq
    global ttech
    global ftech
    
    likelihoodW_Equity = {} 
    likelihoodW_Technology = {} 
    
    for w in test:
        likelihoodW_Equity[w] =  (count_W_inList(w,bagEquityFreqDist) + 1) / denominatorEq
    for w in test:
        likelihoodW_Technology[w] = (count_W_inList(w,bagTechnologyFreqDist) + 1) / denominatorTech

    for w in test:
        finalProbEq *=  likelihoodW_Equity[w]
    print("Prob Equity = " + str(finalProbEq) )
    
    for w in test:
        finalProbTech *= likelihoodW_Technology[w]
    print("Prob Technology = " + str(finalProbTech) )
    
    if finalProbEq > finalProbTech:
        print("\nModel predicts the test query belongs in the EQUITY class")
        if string == "equity":
            teq += 1
        else:
            feq += 1
    else:
        print("\nModel predicts the test query belongs in the TECHNOLOGY class")
        if string == "technology":
            ttech += 1
        else:
            ftech += 1

newEquity = []
for s in AboutEquity:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize(s)
    
    for w in noPunct:
        if w.lower() not in defaultStopwords:
            new += w.lower()
            new += ' '
    newEquity.append(new)

newTechnology = []
for s in AboutTechnology:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize(s)
    
    for w in noPunct:
        if w.lower() not in defaultStopwords:
            new += w.lower()
            new += ' '
    newTechnology.append(new)

for i in range(10):
    
    crossVal(newEquity)
    crossVal(newTechnology)
    
    V = set()  # the vocabulary
    
    bagEquity= []
    for s in trainingSetE:
        tokens = nltk.word_tokenize(s)
        for w in tokens:
            V.add(w)
            bagEquity.append(w)
            bagEquityFreqDist[w] += 1
    
    bagTechnology = []        
    for s in trainingSetT:
        tokens = nltk.word_tokenize(s)
        for w in tokens:
            V.add(w)
            bagTechnology.append(w)
            bagTechnologyFreqDist[w] += 1
            
    total = len(bagEquity) + len(bagTechnology)
    ProbEquity = len(bagEquity)/total
    ProbTechnology  = len(bagTechnology)/total

    #Assign our test string indexes to a string variable
    fullTestList1 = nltk.word_tokenize(testDataE[0])
    fullTestList2 = nltk.word_tokenize(testDataE[1])
    fullTestList3 = nltk.word_tokenize(testDataE[2])

    fullTestList4 = nltk.word_tokenize(testDataT[0])
    fullTestList5 = nltk.word_tokenize(testDataT[1])

    #Check how many values are in vocab
    test_1 = [ w for w in fullTestList1 if w in V]  
    test_2 = [ w for w in fullTestList2 if w in V]
    test_3 = [ w for w in fullTestList3 if w in V]

    test_4 = [ w for w in fullTestList4 if w in V]
    test_5 = [ w for w in fullTestList5 if w in V]

    #Create denominator for Equity
    denominatorEq = 0
    for w in V:
        denominatorEq += (count_W_inList(w, bagEquityFreqDist)+1)

    #Create denominator for Technology
    denominatorTech = 0
    for w in V:
        denominatorTech += ( count_W_inList(w,bagTechnologyFreqDist) + 1)

    print(len(V))
    print(len(trainingSetE))
    print(len(testDataE))
    print(len(trainingSetT))
    print(len(testDataT))

    print("\n\n")

    bayes(ProbEquity, ProbTechnology, test_1, denominatorEq, denominatorTech, "equity")
    bayes(ProbEquity, ProbTechnology, test_2, denominatorEq, denominatorTech, "equity")
    bayes(ProbEquity, ProbTechnology, test_3, denominatorEq, denominatorTech, "equity")

    bayes(ProbEquity, ProbTechnology, test_4, denominatorEq, denominatorTech, "technology")
    bayes(ProbEquity, ProbTechnology, test_5, denominatorEq, denominatorTech, "technology")

    print("\n\n")

    print(teq)
    print(feq)
    print(ttech)
    print(ftech)
    
    precision_eq = teq/(teq+feq)
    precision_tech = ttech/(ttech+ftech)
    avgPrecision_eq += precision_eq
    avgPrecision_tech += precision_tech

    recall_eq = teq/(teq+ftech)
    recall_tech = ttech/(ttech+feq)
    avgRecall_eq += recall_eq
    avgRecall_tech += recall_tech
    
    accuracy = (teq+ttech)/(teq+ttech+feq+ftech)
    print(accuracy)
    avgAccuracy += accuracy
    
    trainingSetE.clear()
    testDataE.clear()
    trainingSetT.clear()
    testDataT.clear()
    
    bagEquityFreqDist.clear()
    bagTechnologyFreqDist.clear()
    
    teq = 0
    feq = 0
    ttech = 0
    ftech = 0
    print("\n\n")

avgPrecision_eq /= 10
avgPrecision_tech /= 10
avgRecall_eq /= 10
avgRecall_tech /= 10
avgAccuracy /= 10

print("\n\n")
print("Average Precision EQ: {}".format(avgPrecision_eq))
print("Average Precision TECH: {}".format(avgPrecision_tech))
print("Average Recall EQ: {}".format(avgRecall_eq))
print("Average Recall TECH: {}".format(avgRecall_tech))
print("Average Accuracy: " + str(avgAccuracy))
print("\n")
print("ProbTechnology: " + str(ProbTechnology))
print("ProbEquity: " + str(ProbEquity))
