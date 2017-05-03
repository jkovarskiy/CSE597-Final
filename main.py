from timeit import default_timer
from sklearn import svm
import nltk
import random
import string
import gensim
import numpy

#Reads in tweets 
def readTweets(path, lbl):
    tweets = []
    tempInput = ""
    transTable = str.maketrans({key: None for key in string.punctuation.replace('#','')}) #Table for removing punctuations except hashtag
    with open(path, encoding="utf-8", mode = "r") as set:
        for line in set:
            if "<end>" in line.lower(): #Checks for tag indicating the end of the tweet
                toks = nltk.word_tokenize(tempInput.translate(transTable)) #tokenizes the tweet
                tweets.append(list(nltk.ngrams(toks,2)))    #appends n-grams 
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
readSet1 = "noisyText_#superhero.txt"
readSet2 = "noisyText_#fyre.txt"
readSet3 = "noisyText_#climate.txt"

numInputs = 0
input += readTweets(readSet1, 1)
label += ([1] * (len(input)-numInputs))
numInputs = len(input)
print("set 1 read")

input += readTweets(readSet2, 2)
label += ([2] * (len(input)-numInputs))
numInputs = len(input)
print("set 2 read")

input += readTweets(readSet3, 3)
label += ([3] * (len(input)-numInputs))
index = list(range(len(input)))
print("set 3 read")

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

#Resizes the data to avoid SVM errors
vSize = 0
for mem in randInput:
    if len(mem) > vSize:
        vSize = len(mem)
        
for idx, mem in enumerate(randInput):
    while len(mem) < vSize:
        mem.append(('<FILL>','<FILL>'))
    randInput[idx] = mem

print("lengths adjusted")

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

wordVecs = []
for idx in range(0, len(randInput)):
    tempVec = model.wv[randInput[idx]]
    dims = tempVec.shape
    wordVecs.append(numpy.reshape(tempVec, dims[0]*dims[1])) #Reshapes embedding for SVM
    
    
print("word vectors built")

#SVM
split = 9700 #training/test split

classify = svm.SVC(gamma = (1/310)) 
classify.fit(wordVecs[0:split], randLabel[0:split])

print("SVM trained")

test = classify.predict(wordVecs[split:dataLim])
results = (test == randLabel[split:dataLim])

#Calculates accuracies
testAcc = 0
for mem in results:
    if mem:
        testAcc += 1
testAcc /= dataLim
    
print("Test accuracy: "+str(testAcc)+"%")

train = classify.predict(wordVecs[0:split])
results = (train == randLabel[0:split])

trainAcc = 0
for mem in results:
    if mem:
        trainAcc += 1
trainAcc /= split
        
print("Training accuracy: "+str(trainAcc)+"%")

endTime = default_timer()

print(str(endTime-startTime)+" seconds elapsed!")



