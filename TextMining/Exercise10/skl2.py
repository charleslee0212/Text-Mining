# skl2.py
# Logistic Regression using SCIKIT Learn

'''
Using vectorizer:
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix


verbose = False
posDocs = []
negDocs = []

# read in posTrain.txt => file of postive training documents
fpPosTrain = open('posTrain.txt', 'r') 
for line in fpPosTrain:
	posDocs.append(line)

# read in, tokenize, and remove stop words for negTrain.txt => file of negative documents 
fpNegTrain = open('negTrain.txt', 'r') 
for row in fpNegTrain:
	negDocs.append(row)

# create lists of labels for the documents
labelPosDocs = [1 for i in posDocs]
labelNegDocs = [0 for i in negDocs]

# create one large list of the docs (order does not matter)
docs = posDocs + negDocs

# create one large list of the labels (order should match the docs)
labels = labelPosDocs + labelNegDocs


# create the feature vectors using CountVectorizer
cv = CountVectorizer(binary=False,max_df=0.95)
cv.fit_transform(docs)
counts = cv.transform(docs)   # counts is now a list of feature vectors

# printing out stuff to see what is going on "under the hood"
# print(cv.vocabulary_)
# print("counts[0] = " + str(counts[0]))
# print("counts[1] = " + str(counts[1]))
# print("counts[2] = " + str(counts[2]))


'''
print("counts[:10] = ")
for i in range(10):
	print("\ni = " + str(i))
	print(counts[i])
'''

# split into training and test sets and labels, y_ are the labels
x_train, x_test, y_train, y_test  = train_test_split( counts, labels,  test_size=0.1, random_state=1)

print("x_train.shape:")
print(x_train.shape)

print("x_test.shape:")
print(x_test.shape)

print("len(y_train):")
print(len(y_train))

print("len(y_test):")
print(len(y_test))

#print("x_train.shape = " + str(x_train.shape))
#print("y_train.shape = " + str(y_train.shape))
# print("y_test.shape = " + str(y_test.shape))
# print("x_test.shape = " + str(x_test.shape))


print("\n\n")
print("Training a Logistic Regression Model...")
scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=5000)

model = scikit_log_reg.fit(x_train,y_train)  # this runs the model on the training set

# get the probabilities for each text document
probs = model.predict_proba(x_test)


# get the predicted values for each test document
y_predicted = model.predict(x_test)

print("x_test.shape = ")
print(x_test.shape)

print("len(y_predicted) = ")
print(str(len(y_predicted)))

print("y_predicted = ")
print(y_predicted)



print("y_predicted full = ")
counter = 0
for i in y_predicted:
	counter += 1
	print(i,end=" ")
	if ((counter % 40 ) == 0):
		print("")



print("\n")
truePos = falsePos = trueNeg = falseNeg = 0
for i in range(len(probs)):
	# uncomment next line if want to print out each test document's tag and resultant classification 
	# print(str(y_test[i]) + ": " + str(probs[i]))
	if (y_test[i] == 0):
		if (probs[i][0] < 0.5):
			# uncomment if want to print out info about documents classified differently than tag
			'''
			print("preceeding is wrong")
			print("x_test[i] = " )
			print(x_test[i])
			print('docs[i] = ' + str(docs[i]))
			'''
			falsePos += 1
		else:
			trueNeg += 1
	if (y_test[i] == 1):
		if (probs[i][0] > 0.5):
			# uncomment if want to print out info about documents classified differently than tag
			'''
			print("preceeding is wrong")
			print("x_test[i] = " )
			print(x_test[i])
			print('docs[i] = ' + str(docs[i]))
			'''
			falseNeg += 1
		else:
			truePos += 1
print("truePos = " + str(truePos))
print("trueNeg = " + str(trueNeg))
print("falsePos = " + str(falsePos))
print("falseNeg = " + str(falseNeg))
	
# look - you can get the true/false positives/negatives in one function call!
print(confusion_matrix(y_test, y_predicted))

precision_positive = truePos/(truePos+falsePos)
precision_negative = trueNeg/(trueNeg+falseNeg)

recall_positive = truePos/(truePos+falseNeg)
recall_negative = trueNeg/(trueNeg+falseNeg)

print("accuracy = " + str( (truePos+trueNeg) / (truePos+trueNeg+falsePos+falseNeg)))
print("precision positive = {}".format(precision_positive))
print("precision negative = {}".format(precision_negative))
print("recall positive = {}".format(recall_positive))
print("recall negative = {}".format(recall_negative))

