'''
Created on 20-Mar-2014

@author: Abhinav
'''


import math
import os
import nltk
import nltk.data
from  nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader


class Lang_Model_Classifier:
    
    
    #Frequency distribution
    def freq_dst(self,posCorpus,negCorpus):
         
        #Creates frequency distribution for words in corpus
        posFreqDist = FreqDist()
        for word in posCorpus.words():
            posFreqDist.inc(word)

        negFreqDist = FreqDist()
        for word in negCorpus.words():
            negFreqDist.inc(word)
 
        #Frequency Distributions with Laplace Smoothing 
        global posLapFreq
        posLapFreq = nltk.probability.LaplaceProbDist(posFreqDist) 
        global negLapFreq
        negLapFreq = nltk.probability.LaplaceProbDist(negFreqDist)

        #GetBigrams
        posBigrams = nltk.bigrams(posCorpus.words())
        negBigrams = nltk.bigrams(negCorpus.words())

        #Get no. of words per corpus
        posWordLen = len(posCorpus.words())
        negWordLen = len(negCorpus.words())


        #FreqDist for Bigrams
        global posBiFreq
        posBiFreq = nltk.probability.LaplaceProbDist(nltk.FreqDist(posBigrams))
        global negBiFreq
        negBiFreq = nltk.probability.LaplaceProbDist(nltk.FreqDist(negBigrams))
        
    # Function to calculate Perplexity, pass a list of words and a Laplace Prob Dist, return a float value
    def perplex(self,testSet, freqDist):

        prob = 0
        for word in testSet:
            prob += math.log(float(1)/freqDist.prob(word))
            
        return math.pow(prob, float(1)/len(testSet))



    # Functions which return if document is Negative or Positive, pass a list of words, get string 
    #For Unigram
    def perp(self,ws):
        pos = self.perplex(ws, posLapFreq)
        print pos 
        neg =  self.perplex(ws, negLapFreq)
        print neg 
        if pos < neg:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'
    
    #For Bigram
    def perpBi(self,ws):
        pos = self.perplex(ws, posBiFreq)
        print pos 
        neg =  self.perplex(ws, negBiFreq)
        print neg 
        if pos < neg:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'


#Main() starts here
def main():
    
    # Corpus Location
    #for training data
    posTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_train'
    negTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_train'

    #for test data
    posTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_test'
    negTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_test'

    # Create Plain Text Corpus for training data
    posCorpus = PlaintextCorpusReader(posTrainCorpus, '.*')
    negCorpus = PlaintextCorpusReader(negTrainCorpus, '.*')


    # Create Plain Text Corpus for test data
    posTstCorpus = PlaintextCorpusReader(posTestCorpus, '.*')
    negTstCorpus = PlaintextCorpusReader(negTestCorpus, '.*')
    
    #GetBigrams
    posBigrams = nltk.bigrams(posCorpus.words())
    negBigrams = nltk.bigrams(negCorpus.words())

    #Get no. of words per corpus
    posWordLen = len(posCorpus.words())
    negWordLen = len(negCorpus.words())
    
    # Creating object of Lang_Model_classifier
    obj1 = Lang_Model_Classifier()
    obj1.freq_dst(posCorpus, negCorpus)
    
    #For negative test data
    for filename in os.listdir(negTestCorpus):
        wordSet =  negTstCorpus.words(filename)
    
        print '**Unigram**'
        unigr = obj1.perp(wordSet)
    
        print unigr
    
        print '**Bigram**'
        bigr = obj1.perpBi(nltk.bigrams(wordSet))
    
        print bigr
        
    #For positive test data    
    for filename in os.listdir(posTestCorpus):
        wordSet2 =  posTstCorpus.words(filename)
    
        print '**Unigram**'
        posunigr = obj1.perp(wordSet2)
    
        print posunigr
    
        print '**Bigram**'
        posbigr = obj1.perpBi(nltk.bigrams(wordSet2))
    
        print posbigr
        
        

# Calling main()
if __name__ == "__main__":
    main()

