# this example uses NaiveBayes, specifical MultiNomial (but only two classifications 0 & 1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix

import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
defaultStopwords = stopwords.words('english')   # can get other languages




# the hand-coded test documetns are coded out, instead will read in from files below
'''
negDocs =  [
"Oh elevenses, i love your pants..the fabrics, the details, the length. ive found elevenses to be classic work pants. reliable, professional, well made. these are none of those things. the cardinal trousers are thin and the rise feels much higher thats it appears in the picture. the red pair are thin, stretchy and cheap. both pairs have to go back. im truly sad.",
"I loved this blouse when i got it and wore it before washing. it fit really well and was flattering. the only time i laundered it i hand- washed it in cold water and hung it to dry. the blouse shrunk at least a full size and in awkward places. the sleeves are now tight and way too short. the overall length shortened by at least 2 inches. the top is so tight in the bust now that the buttons popped open. im very disappointed and surprised.i will be returning it as it is unwearable as is.",
"I wish i had read all the reviews before purchasing. the sweater looked liked everything i would be interested in online. once i pulled it out of the bag, my first thought was how could retailer send me a used, washed and shrunken sweater. it is nothing like the picture. it fits like a midriff top, not exaggerating. im 5.6, ordered the m, dont waste your shipping money on this.",
"The beadwork is gorgeous, but the sleeves are so puffy, it looks as though youre wearing shoulder pads. the fabric of the shirt isnt that fabulous either.",
"Cute and very soft, but the elastic wasnt sewn in right. i took them out of the wash and the waistband was a twisted mess. disappointing for more expensive loungewear.",
"Cute skirt but i agree with the other reviewer. you cant get it on! i tried my skinny daughter tried no go!",
"I was excited to receive this top. it looked great online, vibrant colors with the beautiful detail on the sleeves. when i tried it on, the fabric looked and felt cheap. it is not a flowy top. do not recommend it.",
"This is quite an odd dress. it does wrap and is beautiful material. however, the dress does not close at all in the chest area. i thought perhaps it was meant to be a kimono. it will go back for sure.",
"I love the way these shorts fit and look; very comfy, but after laundering one time, the lavender color became blotchy. used regular detergent in gentle cycle and it looks like someone poured bleach on them in random areas & they are unwearable! i have several colors, but the only color that was a problem is the lavender. $58 for one wear was a waste. i hate to leave negative feedback, but be careful with laundering.",
"No. this is one of the most unflattering things ive ever put on my body. im a very well-proportioned hourglass - i tend to wear a small despite a booty and 36d chest - this made me look huge. dont do it.",
"I loved this shirt until the first time i washed it. it shrunk so much it became unwearable. when i returned it the salesperson said she had also bought this shirt and the same thing happened.",
"Loved the look, fit and styling of this turtleneck. but dry clean only, you have to be kidding?",
"I bought these on sale and they quickly went back. i was very disappointed after reading the reviews. i thought for sure these would be a winner. however, i was sadly mistaken. i ordered another pair of jeans, the same, from a different brand. these were way too small while my others were too large. they didnt live up to my expectation and were still rather long. even though i bought these on sale, i was not happy with the quality.",
"I purchased this swimsuit in the mint. gorgeous color combination. im 5 5  and weigh 117 lbs. i purchased a small. the fit is true to size. i love the front design; however, my husband promptly brought the back of the swimsuit to my attention. the chevron was off center. the design should have been centered with my back; however, it was on the right side of my back. i immediately called retailer assuming the product was defective. the item was sold out on line; however, a supervisor was able to l",
"I am 5 6 and 130 pounds and ordered this in an xs. the sweater is enormous. the material is also scratchy and not soft at all. im disappointed with retailers sizing this year. ive had to return a # of xs items because they are huge. really disappointing.",
"I wanted this dress so much, i ordered it twice. both times the zipper was defective--it would get stuck midway up, at the waistline--so i had to send it back (there isnt an retailer in my area). this is a shame because its got great colors and looks flattering on, and i was looking forward to pairing it with a cardigan/boots for fall. i even thought about taking it to a seamstress to have the zipper replaced, but the cost didnt seem worth it. too bad, as it seems like it would be a nice",
"Beautiful fabric with soft drape. unfortunately the captain kangaroo pockets are awful. unless you have a straight up and down shape with no hips this is very unflattering. the pockets look like saddle bags and add a huge amount of fabric where most people dont want added bulk. very disappointing. returned.",
"I do not understand all the great reviews but that goes to show everyone likes different things! the jacket looks great in the pictures but in person, it just looks awful! gray is one of my favorite colors but this color is so drab and the material looks old and worn. it definitely has a grunge look to it and overall, it is just a wrinkly mess. another reviewer said it looks like old clothes and i have to agree. it looks very shabby. it also reminds me of those smocks they have at the hair salon",
"I was really excited for this jacket to arrive but its going back. because the fabric is coated, it does not drape or hang nicely at all. i expected it to be boxy but it sticks out so oddly in every direction that it looks horrible. im usually an xs and ordered an xs and it looks ridiculous on me.",
"I read the cleaning instructions label carefully. followed them closely. and still, the dye ran and stained some of my colored clothes. pi**ed me off! just after the first wash and even before wearing them, i have streaks of uneven color on the pants. bad quality material or dye is what causes this. now i have to go and see if i can even get out my other stained clothes. thanks a lot retailer."
]


posDocs = [
"Love all the colors in this skirt and that i can wear it with a tee and flat sandals or a black jacket and heels. easy piece to wear many ways. great quality too.",
"This tunic trumped any other i have seen this season. the style, with the delicate open stitchwork around the upper chest gave it quite a feminine appeal. i especially love the weight of the fabric being on the light side. wont have to worry about hot flashes like when wearing a thicker fabric! its warm without being bulky. and to top it off, it was on sale and i was able to grab two colors. this tunic is also age appropriate and flattering for most anyone. extraordinary to say the least.",
"I needed a dress that was easy to throw on for summer days and this dress is perfect for that. its flattering, light weight and unique. ive received a handful of compliments while wearing this dress. i am 5  6  150 lbs, hourglass figure and typically purchase a small or medium (8-10) and chose a small for this dress. id say its still loose on me- which is what i prefer. the scoop neck and cut out allows you wear a normal bra. however, ive noticed im more comfortable wearing a camisole or s",
"I love paige brand pants-they are soft, comfortable, and forgiving. i love these, and want them badly. the are still tight all the way to the knee and then go out into a flattering flare-it is difficult to find the perfect fit on something like this-and paige has done a wonderful job for my body.  my store does not carry petite so i tried these on in regular length. they were significantly too long for me (5 3) probably 3-4 inches to long. i am hoping for a sale so i can buy them in petite for",
"This top is really pretty and nice quality. runs big - i went down a size, and its perfect. coloring is more subtle in person than in the photo.",
"I tried on the petite size in my usual xs, adn i actually have to go down to xxs, i looked overtaken by the shirt. im 5 2  and 115lbs)  cut is flowy and not close to the body, sleeves are narrower, but still ok with athletic built.  color: light one is great for gals with darker complexion and hair, but for my pale self, the darker one was better... but cant go wrong, i ordered both colors and liked both of them.  ruffle is a great addition too...",
"I tried these on on a whim because i liked the shirt that they were displayed with in the store and was surprised how much i liked them! they are a great lighter weight alternative to the pilcro hyphen chino. great for hot days of summer. the subtle vertical stripes go with everything and help elongate the leg. overall a really flattering cut. the waist is not too low and does not create muffin top.",
"These jeans! i tried these on, in addition to the high rise paige denim, and these won out hands down. classic flattering fit from mother, with an element of edginess with the frayed hem. these are long enough on me (im 55) to cuff at the ankle if i dont want a distressed look on a particular day. they are slightly stretchy like other mother denim but not so much that i would size down. i have muscular calves and thighs, but someone who does not could likely size down and be happy with the a",
"I ordered this top in my usual size and am exchanging it for one size smaller. it runs very generous, and so the sizing is a little off. the style and quality are beautiful, so i am anxious to receive the smaller size.",
"The blush stripes are subtle but they definitely give elongating effect to your legs.  very comfortable pair of crop pants but my calves are definitely feeling tight in there!",
"I got a small mauve. the fit is great and the length is perfect for me, just few inches above my knees. cute and cozy! what more can i aske for!?",
"Fun detail with the beading and lace! arms are a little longer while the body of the sweatshirt is a little shorter than expected, but thats the style of the piece. the fit was tts with those proportions mind.  the ladies at the store said that if i ordered the size up, it might be a little longer in the body, but that the arms and shoulders would have been the biggest change. the material isnt too thick, so its a nice lighter layer for fall/spring. really love it!",
"I went ahead and ordered a size up based on previous reviews, but i should have ordered my own size, as theyre a bit loose around the waist. the pants are adorable and the pinstripes very flattering, so i definitely recommend them!",
"This is exactly what i was expecting. cute, comfortable and casual. there are some gold sequins in the scroll work that i didnt see online. they are super pretty in person.",
"I love these pants. i have worn them a number of times already this season. i am 5 so i did have to have them hemmed. i lost the bottom button in the process but there are still 3 or 4 on the pants so i dont think they look odd. i also wear very high boots with these pants so that helps. a crisp white blouse and black leather jacket and i felt like a million bucks!",
"I love this tunic the natural color is just that, this is a tunic so the fit is a little large. i kept it and had it altered because i really do love this top",
"These cropped pants are very light weight and super cute. they seem to run just a bit small (i sized up one size from my usual) and dont seem to stretch so a size larger than you generally take may be necessary. the thin pin stripe design is very light in color so they are quite versatile.",
"I have this dress on today in white and i am coming back to buy the second color even though pink is not my favorite. great comfy, casual dress that pairs well with a variety of shoes and jewelry to dress it up. highly recommend for summer!",
"Size down! i love this item. it goes perfect with leggings but if you are typically a small you would need to order an extra small and so forth. hopefully once i wash this it will shrink some.",
"I purchased this top in a regular small and surprisingly, it fits me very well (im 5 2 , 34b, 26 waist, 36 hips). the hem falls about two inches longer than shown on the model. i like the v-neck the most because the ruffles are not too much, and its not too low cut. i purchased the white color because i think the pattern is unique and its brighter for spring/summer. i think skinny jeans and ankle boots or wedges would make the look very stylish. by the way, the fabric is super soft (but not w"
]

'''


verbose = False
tempPosDocs = []
tempNegDocs = []

# read in, tokenize, and remove stop words for posTrain.txt => file of postive training documents
fpPosTrain = open('posTrain.txt', 'r') 
for line in fpPosTrain:
	tempPosDocs.append(line)
# Now remove all stopwords
posDocs = []
for s in tempPosDocs:
	new = ''
	tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
	noPunct = tokenizer.tokenize( s )
	for word in noPunct:
		if word.lower() not in defaultStopwords:    
			new += word.lower()
			new += ' '
	posDocs.append(new)
if verbose:
	print("\n\nposDocs = ")
	print(posDocs)
	print("\n\n")

# read in, tokenize, and remove stop words for negTrain.txt => file of negative documents 
fpNegTrain = open('negTrain.txt', 'r') 
for row in fpNegTrain:
	tempNegDocs.append(row)
# Now remove all stopwords
negDocs = []
for s in tempNegDocs:
	new = ''
	tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
	noPunct = tokenizer.tokenize( s )
	for word in noPunct:
		if word.lower() not in defaultStopwords:    
			new += word.lower()
			new += ' '
	negDocs.append(new)
if verbose:
	print("\n\nnegDocs = ")
	print(negDocs)
	print("\n\n")



trainData = posDocs
labels = [1 for i in trainData]

trainData2 = negDocs
labels2 = [0 for i in trainData2]
trainData = trainData + trainData2
labels = labels + labels2

count_vect = CountVectorizer()
counts = count_vect.fit_transform(trainData)


# process positive docs
x_train, x_test, y_train, y_test  = train_test_split( counts, labels,  test_size=0.1, random_state=69)

print("\nx_train = ")
print(x_train)
print("\nx_test = ")
print(x_test)
print("\ny_train = ")
print(y_train)
print("\ny_test = ")
print(y_test)


print("\n\n")
print("Training a NB Model...")
model = MultinomialNB().fit(x_train,y_train)
y_predicted = model.predict(x_test)
print('y_predicted = ')
print(y_predicted)
print(np.mean(y_predicted == y_test))
print(confusion_matrix(y_predicted, y_test))






print("\n\nprobs = " )
probs = model.predict_proba(x_test)
# probs = model.predict_proba(test_feature_set)
# probs = model.predict([[0,1]])

truePos = falsePos = trueNeg = falseNeg = 0
for i in range(len(probs)):
	print(str(y_test[i]) + ": " + str(probs[i]))
	if (y_test[i] == 0):
		if (probs[i][0] < 0.5):
			print("preceeding is wrong, wrong, wrong, wrong")
			falsePos += 1
		else:
			trueNeg += 1
	if (y_test[i] == 1):
		if (probs[i][0] > 0.5):
			print("preceeding is wrong, wrong, wrong, wrong")
			falseNeg += 1
		else:
			truePos += 1
print("truePos = " + str(truePos))
print("trueNeg = " + str(trueNeg))
print("falsePos = " + str(falsePos))
print("falseNeg = " + str(falseNeg))

print("accuracy = " + str(  (truePos + trueNeg) /  (truePos+trueNeg+falsePos+falseNeg) ))
