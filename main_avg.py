from timeit import default_timer
from sklearn import svm
import nltk
import random
import string
import gensim
import numpy

#Reads in tweets 
def readTweets(path):
    tweets = []
    tempInput = ""
    transTable = str.maketrans({key: None for key in string.punctuation.replace('#','')}) #Table for removing punctuations except hashtag
    with open(path, encoding="utf-8", mode = "r") as set:
        for line in set:
            if "<end>" in line.lower(): #Checks for tag indicating the end of the tweet
                toks = nltk.word_tokenize(tempInput.translate(transTable)) #tokenizes the tweet
                tweets.append(list(nltk.ngrams(toks,1)))    #appends n-grams 
                tempInput = ""
            else:
                tempInput = tempInput + " " + line.lower()
        else:
            set.close() 
            
    return tweets

                
startTime = default_timer()

input = [] #tweets
label = [] #tweet label
index = [] #tweet index

#Format text data
readSet1 = "taglessText_#superhero.txt"
readSet2 = "taglessText_#fyre.txt"
readSet3 = "taglessText_#climate.txt"
readSet4 = "tweetText.txt"

numInputs = 0
input += readTweets(readSet1)
label += ([1] * (len(input)-numInputs))
numInputs = len(input)
print("set 1 read")

input += readTweets(readSet2)
label += ([2] * (len(input)-numInputs))
numInputs = len(input)
print("set 2 read")

input += readTweets(readSet3)
label += ([3] * (len(input)-numInputs))
numInputs = len(input)
print("set 3 read")

thing = readTweets(readSet4)
input += thing[30000:60000]
del thing
label += ([4] * (len(input)-numInputs))
index = list(range(len(input)))
print("set 4 read")

#Randomize inputs
random.shuffle(index)

randInput = []
randLabel = []
for mem in index:
        randInput.append(input[mem])
        randLabel.append(label[mem])
    
del input
del label

print("input randomized")

#Combines tuples
for idx, mem in enumerate(randInput):
    randInput[idx] = [' '.join(grams) for grams in mem]
    
#Removes empty list elements
while randInput.count([]) > 0:
    idx = randInput.index([])
    del randInput[idx]
    del randLabel[idx]
    
dataLim = len(randLabel) #number of tweets

#Build word vectors
model = gensim.models.Word2Vec(randInput, min_count = 1, sg = 1)

#Averages word vectors for each tweet
wordVecs = []
for idx in range(0, dataLim):
    tempVec = model.wv[randInput[idx]][0]
    for cnt in range(1, len(model.wv[randInput[idx]])):
        tempVec += model.wv[randInput[idx]][cnt]
    tempVec /= len(model.wv[randInput[idx]])
    wordVecs.append(tempVec)
    
    
print("word vectors built")

#SVM
split = 12600 #training/test split

classify = svm.SVC(gamma = (1/10)) 
classify.fit(wordVecs[0:split], randLabel[0:split])

print("SVM trained")

test = classify.predict(wordVecs[split:dataLim])
results = (test == randLabel[split:dataLim])

#Calculates accuracies
testAcc = 0
for mem in results:
    if mem:
        testAcc += 1
testAcc /= len(results)
    
print("Overall test accuracy: "+str(testAcc)+"%")

testAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx+split]==1):
        testAcc += 1
testAcc /= randLabel[split:dataLim].count(1)

print("#NationalSuperheroDay test accuracy: "+str(testAcc)+"%")       

testAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx+split]==2):
        testAcc += 1
testAcc /= randLabel[split:dataLim].count(2)

print("#FyreFestival test accuracy: "+str(testAcc)+"%") 

testAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx+split]==3):
        testAcc += 1
testAcc /= randLabel[split:dataLim].count(3)

print("#ClimateMarch test accuracy: "+str(testAcc)+"%")

testAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx+split]==4):
        testAcc += 1
testAcc /= randLabel[split:dataLim].count(4)

print("Random tweet test accuracy: "+str(testAcc)+"%")
    
   
train = classify.predict(wordVecs[0:split])
results = (train == randLabel[0:split])

trainAcc = 0
for mem in results:
    if mem:
        trainAcc += 1
trainAcc /= split
        
print("Overall training accuracy: "+str(trainAcc)+"%")

trainAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx]==1):
        testAcc += 1
testAcc /= randLabel[0:split].count(1)

print("#NationalSuperheroDay training accuracy: "+str(testAcc)+"%")

trainAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx]==2):
        testAcc += 1
testAcc /= randLabel[0:split].count(2)

print("#FyreFestival training accuracy: "+str(testAcc)+"%")

trainAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx]==3):
        testAcc += 1
testAcc /= randLabel[0:split].count(3)

print("#ClimateMarch training accuracy: "+str(testAcc)+"%")

trainAcc = 0
for idx in range(0,len(results)):
    if (results[idx]==1) and (randLabel[idx]==4):
        testAcc += 1
testAcc /= randLabel[0:split].count(4)

print("Random tweet training accuracy: "+str(testAcc)+"%")

endTime = default_timer()

print(str(endTime-startTime)+" seconds elapsed!")



