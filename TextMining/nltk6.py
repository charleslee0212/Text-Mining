#
# IMPORTANT
# IMPORTANT
# IMPORTANT: Before looking at this code do the example in section 
# 4.3 by hand (with paper/pencil....)
#
#
# This code multiples the likelihoods together like in equation 4.9 in Jurafsky
# It uses the exact same example as in section 4.3 of Jurafsky

import nltk
from nltk.corpus import wordnet
from collections import defaultdict


# helper function to count how many times a word, w,  occurs in a list, aList
def count_W_inList(w, aList):
	count = 0
	for w2 in aList:
		if w2 == w:
			count += 1
	return(count)

# positive training documents
AllPos = [

"After this assignment, I now think that engineers truly are responsible for diversifying their fields. Before, I thought this was more of just a moral issue for engineers. However, especially after the video, I realized that without diversity, the things that engineers produce can literally fail to do their job due to neglect to diversity in the population",

"Definetly, it made put the decisions that engineers make into perspective. One mistake can have a snowball effect similar to how the facial recogontion software the speaker was exposed to could not recognize her face",

"I already knew that it was very important for engineers to think about the diversity of the world when doing there work. However, this assignment showed me that it is more important than I ever thought.",

"I learned that engineers need to consider how technology not only affects them but minorities. I will now take personal responsibility that my work is accessible by all and that race is not a factor in the ability for technology to work for you. I will also be more mindful to reach out to civil right adovcates and society before assuming there is no bias in the projects I am involved in. ",

"I understand now that to be a good engineer you must keep people safe but you also must make sure you do not exclude anyone when creating a product that is meant for everyone",

"I understood that engineers have a high responsibility to the world. I knew that as engineers we should strive to provide people with newer and better technologies. This was the first time I had really considered the sideffects of not being able to accomplish that. If anything, this assignment gave me more awreness as to why engineers have such a high responsibility to the world",

"In a way, yes the assignment did change my view, but not very much. I have always thought that more diverse groups are better. I would never want to be in a group of people just like me because there would be no discussion of the product to make it better. We would all have the same ideas. However, I didn't think of engineers as mediators before. Engineers are responsible for making technology useful for everyone without separating groups because of differences between them",

"It changed my views on the roles and responsibilities of engineers because I realized that its our responsibility to promote diversity and make sure that we never start to discriminate any types of people, even unintentionally. Its our responsibility to identify the inconsistencies in technology, to address those inconsistencies and fix them so that they obey the code of ethics.",

"It didnt change my views, but strengthened them. I think engineers are fully responsible for what they put out into the world, and it should be the most important thing to them to ensure that their product is ready for consumer use. Learning of this situation just makes my feelings toward this stronger, why put out a project that isn't going to work fully. It's like having a car that can only get you halfway there all the time. ",

"It showed me that it is the responsibility of all engineers to keep in mind the diverse population of America when inventing new technologies. This is so important, and it has the potential to keep from creating accidental bias",

"This assignment allowed me to further expand my ideas to consider that the intricate workings of computer programs (such as machine learning) can actually have a noticeable effect on people to whom it doesn't cater. Most individuals are just oblivious to this type of discrimination that does not affect them personally in his or her daily live(s).",

"This assignment reinforced my idea that the more variety of people the more different ideas can be thrown around. ",

"This module taught me that engineers have a ton of responsibility. Engineers must make sure that their products aren't biased in any way. Bias in products can contribute to many social problems so it is important to account for many types of problems that can occur. I never thought about how biases can influence products before this module.",

"Yes, engineers also take on a very important ethical role while developing any sort of new technology. Bias needs to be addressed, and keeping products available and usable by the entire population is just as vital as programming or building it.",

"Yes, it has changed my views about my roles and responsibilities. It has made me aware of the issue of diversity we have at hand and what I need to do to solve all the related problems. It has given me a motivation and a goal to work towards for my career as an engineer, especially as a woman engineer. This assignmnet had also equipped me with the tools I need to promote and enhance diversity in our society both in and out of the field of Engineering.",

"Yes, a little bit. I realize how important it is to spot and get rid of biases. Engineers/Computer Scientists have a responsibility to be inclusive to the entire population. This means ridding biases.",

"Yes, engineers have far greater responsibilities than the basic fundamental cannons.  We have to consider the bigger picture and realize that there is depth in every cannon and we will be held to each one's fullest extent",

"Yes, engineers need to realize that everything that they do can affect other people and they need to strive to make their products or services inclusive for all people",

"Yes, this assignment definitely changed my views on the roles and responsibilities of engineers. I now realize we uphold a responsibility not only in the technical world, but the entire world in general. What we do can affect many people in good and bad ways, and that is something I did not realize until now",

"Yes, what I learned did change how I view the roles of engineers because in most case we are working with people who are different than ourselves and we need to be able accommodate everyone to make sure that their needs are taken care of",

"this assignment added the responsibility to include everyone as an engineer because diversity only helps out the engineering profession" ]


# negative training documents
AllNeg = [

"As explained in problem one, what I learned from this assignment is the responsibility that professional engineers have when developing technology.  They are the ones who have to make decisions that can, and most likely will, have large impacts on many people and society as a whole.  I learned this from reading the articles and making sense of the information presented about the regulations and rules that simply are just not there for the development of new technologies.",

"Honestly, not really. I feel that engineers should always act ethically with every individual in mind, so this kind of just highlighted some mistakes that have been made by engineers",

"I always understood that engineers had many responsibilities, but I had not thought about how people could be excluded from certain products. I suppose I thought more about bridges than coding. It is important to be aware of all the biases that could form because as an engineer you will work with many other engineers in different disciplines. Even though your part of the project is not bias, their part may be. Engineers' work reaches every part of the world, thus their work must benefit every person of the world. ",

"I have always believed that engineers must represent the best in society, which include including diverse ideas and cultures in the workplace to provide best for society.",

"I would not say my views on the responsibilites of engineers changed, as this is something I have always believed engineers should be responsible for. The inclusivity of all types of people should be valued in any job or work environment.",

"It changed my viewpoint on the amount of times a technology should be tested and by whom the technology should be tested by before the technology is released. For example, all different people should test the technology to ensure that the technology accounts for all individuals. Also the technology should be checked multiple times before being released. This will ensure that the technology has few errors. I also think that it is a good idea for the government to regulate the creation and the release of technology, The governemnt should be in charge of testing these products before they are put into the market instead of the company just simply creating and releasing the product.",

"It did not change my view on the roles and responsibilities of engineers. ",

"It didn't really change my overall views of engineers, but it makes me appreciate my technology more. It's crazy to think that a lot of smart phones have this technology readily used and available.",

"My personal ethics are very similar to those in the Code of Ethics. Therefore, my views didn't change much and were therefore reinforced to challenge people who think otherwise.",

"My thoughts have not changed about the responsibilities of engineers. I still believe engineers need to maintain the same integrity after reading this article. I think this article made me realize how important diversity is within the engineering discipline.",

"No the assignment did not change my views on the role of engineers.  I already thought that we as a profession should not be pushing out unfinushed work that would be detrimental to people of this world.",

"No they did not.  I have always understood the importance of working with a diverse group in any situation.  One thing I did learn from this activity was the importance of testing your product before putting it out to the public.  When testing if you find flaws it is imperative that you fix the problems before releasing your product to the public. ",

"This assignment did not change my views on the roles or responsibilites for engineers.  I believe that we should always be considerate of designing and implementing technology that is beneficial and accurate for all ethnicities, races, and genders.",

"What I learned in this assignment didn't change my views on teh roles and responsibilities of engineers.  This is because I have always thought that it is important to get multiple perspectives on a project so that the best possible product can be created.  Inclusion is an important part of an engineer's job."

]

# there are 2 postive and 3 negative in the training
total = len(AllPos) + len(AllNeg)
ProbPos = len(AllPos) / total
ProbNeg  = len(AllNeg) / total

from nltk.corpus import stopwords

defaultStopwords = stopwords.words('english')   # can get other languages

from nltk.tokenize import RegexpTokenizer

# lets remove all stopwords

newAllPos = []

for s in AllPos:

    new = ''

    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator

    noPunct = tokenizer.tokenize( s )

    for word in noPunct:

        if word.lower() not in defaultStopwords:

            new += word.lower()

            new += ' '

    newAllPos.append(new)


newAllNeg = []

for s in AllNeg:

    new = ''

    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator

    noPunct = tokenizer.tokenize( s )

    for word in noPunct:

        if word.lower() not in defaultStopwords:

            new += word.lower()

            new += ' '

    newAllNeg.append(new)


fullTestString = "Yes, it changed my views, since, for example, suppose your design is created with flaws.  Then the rules will maintain bias."
#"Yes, it did.  I now appreciate why we should consider inclusivity and act with professional integrity."
#"Honestly, yes, it did.  This highlighted the clear benefit of considering different cultures." 
#"No, it did not. The computer program can already identify most faces without a mistake."  
#"No. From my perspective this is bogus, we do not discriminate in America" 
#"No. I do not think it moral for software to over accomodate minorities." 
#"It actually did not.  From my perspective this is definitely accidental." 


#NEGATIVE
#"It did not change my view on the roles and responsibilities of engineers. "
#"Honestly, not really. I feel that engineers should always act ethically with every individual in mind, so this kind of just highlighted some mistakes that have been made by engineers"
#POSITIVE
#"It changed my views on the roles and responsibilities of engineers because I realized that its our responsibility to promote diversity and make sure that we never start to discriminate any types of people, even unintentionally. Its our responsibility to identify the inconsistencies in technology, to address those inconsistencies and fix them so that they obey the code of ethics."
#"After this assignment, I now think that engineers truly are responsible for diversifying their fields. Before, I thought this was more of just a moral issue for engineers. However, especially after the video, I realized that without diversity, the things that engineers produce can literally fail to do their job due to neglect to diversity in the population"


# Some variable definitions:

# V is the vocabulary, i.e. set { } of all words (duplicates removed because it is a set)
# C is the set of classes:   C = { CP, CN }, where 
#    CP = class postive, i.e. the class of positve reviews 
#    CN = class negative, i.e. the class of negative reviews 

# bagPos defined as bag of words for positive class documents.  NOTE - a "bag" is a like a set 
# but includes all duplicates In other words, bagPos is the adding together of all postive 
# class contents into on list without removing duplicates.

# bagNeg defined as bag of words for negative class documents.  

# Create bagPos, bagNeg and V

V = set()
bagPos = []   # bagPos: words in "positive" class
for s in newAllPos:
    ts = nltk.word_tokenize(s)
    for w in ts:
        bagPos.append(w)
        V.add(w)

bagNeg = []   # bagNeg: words in "negative" class
for s in newAllNeg:
    ts = nltk.word_tokenize(s)
    for w in ts:
        bagNeg.append(w)
        V.add(w)

print("\nbagPos = ")
print(bagPos)
print("\nbagNeg = ")
print(bagNeg)
print("\nV = ") 
print(V)

fullTestList = nltk.word_tokenize(fullTestString)
test = [ w for w in fullTestList if w in V]     
print("\nTest query = ")
print(test)

likelihoodW_pos = {} # dictionary of probabilities for positive class

# create the denominator for the positive class
# for each word in the complete vocabulary, sum up (count(w,c) + 1)
denominator = 0
for w in V:
	denominator += ( count_W_inList(w,bagPos) + 1)
print("debug: denominator postive class = " + str(denominator))

# for each query word w, get the likelihood[w,postive]
for w in test:
	likelihoodW_pos[w] =  (count_W_inList(w,bagPos) + 1) / denominator
	print("debug: numerator pos class for w = " +  w + "  = " + str(count_W_inList(w,bagPos) + 1))

print("likelihoodW_pos = ")
print(likelihoodW_pos)



likelihoodW_neg = {} # dictionary of probabilities for negative class

# create the denominator for the negative class
# for each word in the complete vocabulary, sum up (count(w,c) + 1)
denominator = 0
for w in V:
	denominator += ( count_W_inList(w,bagNeg) + 1)
print("debug: denominator negative class = " + str(denominator))

# for each query word w, get the likelihood[w,negative]
for w in test:
	likelihoodW_neg[w] =  (count_W_inList(w,bagNeg) + 1) / denominator
	print("debug: numerator negative class for w = " +  w + "  = " + str(count_W_inList(w,bagNeg) + 1))

print("likelihoodW_neg = ")
print(likelihoodW_neg)




# final calculations

# P(-) P(S | -) 
# where S is the test query Sentence, S = "predictable with no fun"
finalProbNeg = ProbNeg 
for w in test:
	finalProbNeg *=  likelihoodW_neg[w]
print("Prob negative = " + str(finalProbNeg) )

# P(+) P(S | +) 
# where S is the test query Sentence, S = "predictable with no fun"
finalProbPos = ProbPos 
for w in test:
	finalProbPos *=  likelihoodW_pos[w]
print("Prob positive = " + str(finalProbPos) )


if (finalProbPos > finalProbNeg):
	print("\nModel predicts the test query belongs in the POSITVE class")
else:
	print("\nModel predicts the test query belongs in the NEGATIVE class")
