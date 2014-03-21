'''
Created on 20-Mar-2014

@author: Abhinav
'''

import math
import nltk
import nltk.data

from  nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader

#Perceptron class implementing training as well as testing of Corpus
class percep_train:
    
    #weight vector
    weigthVector = dict()
    #feature vector
    featureVectors = dict()
    
    #Constructor for initialising weight vector for positive as well as negative corpus(training)
    def __init__(self,posCorpus,negCorpus):
        
        for word in posCorpus.words():
            self.weigthVector[word] = 0       
        for word in negCorpus.words():
            self.weigthVector[word] = 0
            
    # creating feature vector for training corpus
    def train_featvectr(self,posCorpus,negCorpus):
        
        #Get Corpora File IDs
        posFiles = dict.fromkeys(posCorpus.fileids())
        negFiles = dict.fromkeys(negCorpus.fileids())
        #feature vector based on frequency distribution of words
        for trFile in posCorpus.fileids():
            fileFreqDist = FreqDist()
            fileFreqDist = nltk.FreqDist(posCorpus.words(trFile))
            self.featureVectors[trFile] = fileFreqDist
    
    
        for trFile in negCorpus.fileids():
            fileFreqDist = FreqDist()
            fileFreqDist = nltk.FreqDist(negCorpus.words(trFile))
            self.featureVectors[trFile] = fileFreqDist
          
        # training perceptron classifier  
        for x in xrange(10):
    
            for item in self.featureVectors.items():
                freq = item[1]
                tempCount = 0
        
                for word in freq.keys():
                    # A temporary variable which is multiplication of weight vector and feature vector for each word
                    tempCount += self.weigthVector[word] * freq.freq(word)
                # updating weight vector based on value of tempCount
                if tempCount == 0:
                    for word in freq.keys():
                        self.weigthVector[word] = freq.freq(word)
                elif tempCount > 0 and negFiles.has_key(item[0]):
                    # Predicted +ve but actually negative
                    for word in freq.keys():
                        self.weigthVector[word] = self.weigthVector[word] - freq.freq(word)
                elif tempCount < 0 and posFiles.has_key(item[0]):
                    # Predicted -ve but actually positive
                    for word in freq.keys():
                        self.weigthVector[word] = self.weigthVector[word] + freq.freq(word)
            print x,' ',self.weigthVector[word]
       
    # testing corpus feature vector
    def corpusFeatureVectors(self,corpus):
        featVect = dict()
        for trFile in corpus.fileids():
            fileFreqDist = FreqDist()
            fileFreqDist = nltk.FreqDist(corpus.words(trFile))
            featVect[trFile] = fileFreqDist
        return featVect
    
    # testing the test corpus
    def testCorpus(self,corpus, features):  
        for testFile in corpus.fileids():        
            ff = features.get(testFile)
            cc = 0
        
            for word in ff.keys():
                if self.weigthVector.has_key(word):
                    cc += self.weigthVector[word] * ff.freq(word)
            
            if cc>0:
                print testFile + ' POSITIVE'
                

            elif cc<0:
                print testFile + ' NEGATIVE'
                

            else:
                print 'CC is 0'
                

        


    
 
        
# Main function 
 
def main():  
    
     # Corpus Location
     #train data
     posTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_train'
     negTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_train'


     # Create Plain Text Corpus from train data
     posCorpus = PlaintextCorpusReader(posTrainCorpus, '.*')
     negCorpus = PlaintextCorpusReader(negTrainCorpus, '.*')

     
     #Corpus Location
     #test data   
     posTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_test'
     negTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_test'     

         
     # Create Plain text Corpus from Test Corpora
     posTest = PlaintextCorpusReader(posTestCorpus, '.*')
     negTest = PlaintextCorpusReader(negTestCorpus, '.*')
     
     # Creating Object of class percep_train
     obj1 = percep_train(posCorpus,negCorpus)
     
     #Calling for training of classifier
     obj1.train_featvectr(posCorpus, negCorpus)  
  
     # Calling for testing of classifier
     posFeat = obj1.corpusFeatureVectors(posTest)
     negFeat = obj1.corpusFeatureVectors(negTest)


     #Negative test data output
     obj1.testCorpus(negTest, negFeat)

     #Positive test data output
     obj1.testCorpus(posTest, posFeat)

        
# Calling main()
if __name__ == "__main__":
    main()