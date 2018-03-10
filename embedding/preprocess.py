##############
# Zijian Yao #
##############
import os, sys
import numpy as np

wordIdx = {}
punc = ["，", "。", "？", "！", "、", "；", "："]

def readFileToCorpus(f,n):
    """ Reads in the text file f which contains one sentence per line.
    	Return a list of n-grams in the dataset
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        unwanted_chars = "，。\n"
        print("reading file ", f)
        for line in file:
            i += 1
            sentence = line.split("\t")[3] # split the line into a list of words
            for p in punc:
                sentence = sentence.replace(p,"\n")
            for k in range(len(sentence)-n):
            	if '\n' not in sentence[k:k+n]: 
            		corpus.append(sentence[k:k+n])  # extend the current list of words with the words in the sentence
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it 
        file.close()
        return corpus
    else:
        print("Error: corpus file ", f, " does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

def wordSegmentation(wordFrequency, threshold):
	newWords = []
	for word in wordFrequency:
		if wordFrequency[word] >= threshold:
			newWords.append(word)
	return newWords

def wordFrequency(list):
	wordfreq = {}
	for word in list:
		if word not in wordfreq:
			wordfreq[word] = 0 
		wordfreq[word] += 1
	return wordfreq

def computeWordIndex(words):
	for i in range(len(words)):
		wordIdx[words[i]] = i 

def getWordIndex(word):
	return wordIdx[word]

def wordCooccurCount(f, words, windowSize):
    cooccurDic = {}
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        for line in file:
            sentence = line.split("\t")[3]  # split the line into a list of words
            for k in range(len(sentence)-windowSize):
            	if sentence[k] in words: 
            		idx = getWordIndex(sentence[k])
            		for word in sentence[k+1:k+windowSize]:
            			if word in words:
            				if (idx, getWordIndex(word)) not in cooccurDic:
            					cooccurDic[(idx, getWordIndex(word))]=0
            					cooccurDic[(getWordIndex(word),idx)]=0
            				cooccurDic[(idx, getWordIndex(word))]+=1
            				cooccurDic[(getWordIndex(word),idx)]+=1
            		for i in range(1,windowSize-2):
            			word = sentence[k+i:k+i+2]
            			if word in words:
            				if (idx, getWordIndex(word)) not in cooccurDic:
            					cooccurDic[(idx, getWordIndex(word))]=0
            					cooccurDic[(getWordIndex(word),idx)]=0
            				cooccurDic[(idx, getWordIndex(word))]+=1
            				cooccurDic[(getWordIndex(word),idx)]+=1
            	if sentence[k:k+2] in words:
            		idx = getWordIndex(sentence[k:k+2])
            		for word in sentence[k+2:k+windowSize]:
            			if word in words:
            				if (idx, getWordIndex(word)) not in cooccurDic:
            					cooccurDic[(idx, getWordIndex(word))]=0
            					cooccurDic[(getWordIndex(word),idx)]=0
            				cooccurDic[(idx, getWordIndex(word))]+=1
            				cooccurDic[(getWordIndex(word),idx)]+=1
            		for i in range(2,windowSize-2):
            			word = sentence[k+i:k+i+2]
            			if word in words:
            				if (idx, getWordIndex(word)) not in cooccurDic:
            					cooccurDic[(idx, getWordIndex(word))]=0
            					cooccurDic[(getWordIndex(word),idx)]=0
            				cooccurDic[(idx, getWordIndex(word))]+=1
            				cooccurDic[(getWordIndex(word),idx)]+=1
        file.close()
        return cooccurDic
    else:
        print("Error: corpus file ", f, " does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

if __name__ == "__main__":
	inFile = "small.txt"
	posSeeds = ["悠","欢"]
	negSeeds = ["荒","旧"]
	outFile = open("sentiLexicon", "w")
	unigram = readFileToCorpus(inFile, 1)
	unigramFrequency = wordFrequency(unigram)
	bigram = readFileToCorpus(inFile, 2)
	bigramFrequency = wordFrequency(bigram)
	words = wordSegmentation(unigramFrequency,3) + wordSegmentation(bigramFrequency,2)
	print(words)
	computeWordIndex(words)
	cooccurDic = wordCooccurCount(inFile, words, 5)
	#for pair in cooccurDic:
	#	sys.stdout.write("("+words[pair[0]]+', '+words[pair[1]]+"): "+str (cooccurDic[pair])+", ")
	nWords = len(words)
	transMatrix = np.zeros((nWords,nWords))
	for pair in cooccurDic:
		transMatrix[pair[0]][pair[1]] = cooccurDic[pair]
	row_sums = transMatrix.sum(axis=1)
	for pair in cooccurDic:
		transMatrix[pair[0]][pair[1]] = cooccurDic[pair]/row_sums[pair[0]]
	rP = np.ones(nWords)/nWords
	rN = np.ones(nWords)/nWords
	rP0 = np.ones(nWords)/nWords
	rN0 = np.ones(nWords)/nWords
	#posSeeds = ["香","爱","欢","贤","喜","瑞"]
	#negSeeds = ["寒","愁","孤","苦","悲","怨"]
	posP = np.zeros(nWords)
	for word in posSeeds:
		posP[wordIdx[word]] = 1/len(posSeeds)
	negP = np.zeros(nWords)
	for word in negSeeds:
		negP[wordIdx[word]] = 1/len(negSeeds)
	alpha = 0.85
	iter = 10
	for _ in range(iter):
		rP = alpha * np.dot(transMatrix,rP0) + (1 - alpha) * posP
		rN = alpha * np.dot(transMatrix,rN0) + (1 - alpha) * negP
		diffP = np.amax(np.absolute(rP - rP0))
		diffN = np.amax(np.absolute(rN - rN0))
		rP0 = rP
		rN0 = rN
		print("Finished one iteration, diffP = %f, diffN = %f" % (diffP, diffN))
	rS = rP - rN
	sentiLex = []
	for i in range(nWords):
		sentiLex.append((words[i], rS[i]))
	sentiLex.sort(key = lambda x: x[1], reverse = True)
	for i in range(nWords):
		outFile.write("%s\t%f\n" % sentiLex[i])


