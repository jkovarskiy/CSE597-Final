textPath = "tweetText_#climate.txt"
newTextPath = "taglessText_#climate.txt"
writeFile = open(newTextPath,encoding="utf-8",mode="w")
topic = "#climatemarch"

with open(textPath,encoding="utf-8",mode = "r") as tweets:
    for line in tweets:
        newLine =  line.lower()
        newLine = newLine.replace(topic,"")
        writeFile.write(newLine)
    else:
        tweets.close()
        writeFile.close()