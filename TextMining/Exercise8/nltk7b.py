'''
This example introduces code for the cross-entropy loss function.

The example goes on to "tune" the weights vector, w, to minimize cross-entropy loss.
This is not actually "practically useful", but illustrates the key idea of optimization, 
namely that we keep changing the values of the paremeters to minimize the loss

'''

import nltk
from math import exp, log


def zValue(fv,w,b):
	if (len(fv) != len(w)):
		print("ERROR, length mismatch, exiting...")
		exit()
	z = 0
	for i in range(0,len(fv)):
		z += fv[i] * w[i]
	z += b
	return(z)

def sigma(z):
	return( 1.0 / (1 + exp(-z)))

wordsPositive = set()
fpp = open("positive-words.txt",mode='r',encoding="ISO-8859-1")
for line in fpp:
	aStr = line.replace('\n','')
	wordsPositive.add(aStr)
# print("len(wordsPositive) = " + str(len(wordsPositive)))

wordsNegative = set()
fpn = open("negative-words.txt",mode='r',encoding="ISO-8859-1")
for line in fpn:
	aStr = line.replace('\n','')
	line.replace('\n','')
	wordsNegative.add(aStr)
# print("len(wordsNegative) = " + str(len(wordsNegative)))


fullTestString = 'the movie was wonderful, I loved it!  non-linear plot, great acting, enthralling.'
test = nltk.word_tokenize(fullTestString)


# feature vector has three parts:
# fv[0] = number of words in "good lexicon"
# fv[1] = number of words in "bad lexicon"
# fv[2] = length of review  # hypothesis: the longer the review the more negative...

# fv = [12, 4, len(fullTestString)]  # hand coded example

print('test = ' + str(test))
fv = [0,0,0]
for word in test:
	if word in wordsPositive:
		fv[0] += 1
	if word in wordsNegative:
		fv[1] += 1
fv[2] = len(fullTestString)

print("fv = " + str(fv))

w = [1, -2, -0.01]   
b = 0

print('w = ' + str(w))

z = zValue(fv,w,b)
print('z = ' + str(z))

y = sigma(z)
print('y = ' + str(y))

if (y > 0.5):
	print("positive")
else:
	print("negative")


''' ========================

* Want to choose parameters w,b that 'maximize the log probability of the true y labels in the training data given teh observations x.'

*(5.7)  L(y^,y) = How much Y^ differs from the true y

*(5.11)	L_CE(w,b) = -[y log(S(w.x+b)) + (1-y)log(1 - S(w.x+b))]
	              = -[y log(S(z)) + (1-y)log(1 - S(z))]
where y = {0 or 1}, S = sigmoid function, w.x is dot product of w and x, and z = (w.x + b)

This is called the "cross-entropy loss" function

'''

print("\n\nNow trying to optimize w,b for the above example.")
print("Here I just change the weight values, w")
print("The closer the resulant value to zero the better.")

w = [1, -2, -0.01]   
b = 0 
x = fv  # x is our feature vector

# lets try "tuning" of w by just experimenting with 4 values of w
# We see in this hard-coded example, for this ONE test document, that we 
# decreas the cross entropy loss to near zero, thus each version of w is "better"

# Equation 5.11
# LCE_1 means Loss Cross Entropy _ 1 , where the _1 is just the first version, 
#       there will be _2, _3, etc...)


y = 1 # classified as positive - in other words we know the document (test) is positive
# y = 0  # if document were negative
print("zValue(w,x,b) = " + str(zValue(w,x,b)))
LCE_1 = -1 * ( y * log(sigma(zValue(w,x,b))) + (1-y) * log( 1 - sigma(zValue(w,x,b))))
print('\nw = ' + str(w))
print("LCE_1 = " + str(LCE_1))

# lets try new values for w
w = [1, -2, -0.001]
b = 0 
x = fv  # x is our feature vector
LCE_2 = -1 * ( y*log(sigma(zValue(w,x,b))) + (1-y)*log( 1 - sigma(zValue(w,x,b))))
print('\nw = ' + str(w))
print("LCE_2 = " + str(LCE_2))

# lets try new values for w
w = [2, -2, -0.01]
b = 0 
x = fv  # x is our feature vector
LCE_3 = -1 * ( y*log(sigma(zValue(w,x,b))) + (1-y)*log( 1 - sigma(zValue(w,x,b))))
print('\nw = ' + str(w))
print("LCE_3 = " + str(LCE_3))

# lets try new values for w
w = [2, -2, -0.001]
b = 0 
x = fv  # x is our feature vector
LCE_4 = -1 * ( y*log(sigma(zValue(w,x,b))) + (1-y)*log( 1 - sigma(zValue(w,x,b))))
print('\nw = ' + str(w))
print("LCE_4 = " + str(LCE_4))



