import random

noisePath = "#superhero_syn.txt"
textPath = "tweetText_#superhero.txt"
newTextPath = "noisyText_#superhero.txt"
writeFile = open(newTextPath,encoding="utf-8",mode="w")
topic = "#nationalsuperheroday"

synonyms = []

with open(noisePath,"r") as noise:
    for line in noise:
        synonyms.append(line.strip('\n'))
    else:
        noise.close()
        
with open(textPath,encoding="utf-8",mode = "r") as tweets:
    for line in tweets:
        randIdx1 = int(10*random.random())
        randIdx2 = int(10*random.random())
        newLine =  line.lower()
        newLine = newLine.replace(topic,synonyms[randIdx1]+" "+synonyms[10+randIdx2])
        writeFile.write(newLine)
    else:
        tweets.close()
        writeFile.close()
        