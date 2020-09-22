from random import random

# this function has no bearing an reality, its only purpos
# is to take in postive/negative test documents and return "positive" or "negative"
# so you can calculate the various recall, precision, and accuracy measures.

tp = 0
fp = 0
tn = 0
fn = 0

def runModel(s):
    global tp
    global fp
    global tn
    global fn
    rNum = random()
    if('neg' in s):
        if(rNum < 0.3):
            fn += 1
            return('positive')
        else:
            tn += 1
            return('negative')
    if('pos' in s):
        if(rNum < 0.4):
            fp += 1
            return('negative')
        else:
            tp += 1
            return('positve')


posTests = [
'posTest0',
'posTest1',
'posTest2',
'posTest3',
'posTest4',
'posTest5',
'posTest6',
'posTest7',
'posTest8',
'posTest9'
]

negTests = [
'negTest0',
'negTest1',
'negTest2',
'negTest3',
'negTest4',
'negTest5',
'negTest6',
'negTest7',
'negTest8',
'negTest9'
]

print("Running positive tests")
for s in posTests:
	result = runModel(s)
	print(result)
	# do calculation here to keep track of truePostives and falseNegatives

print("Running negative tests")
for s in negTests:
	result = runModel(s)
	print(result)
	# do calculation here to keep track of trueNegatives and falsePositives

# write code to calculate the postiveRecall, postivePrecision, negativeRecall, negativePrecision, and accuracy
# It should calculate these measure for the 20 tests (not once per test, but all of the measure for the 20 tests as a group)

precision_positive = tp/(tp+fp)
precision_negative = tn/(tn+fn)

recall_positive = tp/(tp+fn)
recall_negative = tn/(tn+fp)

accuracy = (tp+tn)/(tp+tn+fp+fn)

print("\nprecision:")
print("Positive System: " + str(precision_positive))
print("Negative System: " + str(precision_negative) + "\n")

print("recall:")
print("Positive System: " + str(recall_positive))
print("Negative System: " + str(recall_negative) + "\n")

print("accuracy: " + str(accuracy))





