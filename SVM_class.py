'''
Created on 20-Mar-2014

@author: Abhinav
'''
import math
import nltk
import nltk.data
from  nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader
import numpy as np
from sklearn import svm


class svm_classifier:
    
    #Frequency Distribution
    trainFreq = FreqDist()
    #noFeat = 0
    #trainKeys = []
    
    
    def __init__(self,posCorpus,negCorpus):
        
        #Create Frequency Distribution from both Positive and Negative Corpora
        self.trainFreq = nltk.FreqDist(posCorpus.words() + negCorpus.words())
        

        
    def featureList(self,corpus):
        
        #No of Features
        noFeat = len(self.trainFreq)
        
        #Get Keys to maintain Order
        trainKeys = self.trainFreq.keys()
        
        #Generate feature vector
        featList = []
        for trFile in corpus.fileids():
            listItem = [0]*noFeat
            fileFreqDist = FreqDist()
            fileFreqDist = nltk.FreqDist(corpus.words(trFile))
        
            i =0
            for key in trainKeys:
                if fileFreqDist.has_key(key):
                    listItem[i] = fileFreqDist.get(key)
                    i=i+1
            
            featList.append(listItem)
        
        return featList


def main():
    
    
    # Corpus Location
    #train data
    posTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_train'
    negTrainCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_train'

    # Corpus Location
    #test data
    posTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/pos_test'
    negTestCorpus = 'C:/Users/Abhinav/Desktop/Course work/NLP/txt_sentoken/neg_test'  

    
    
    # Create Plain Text Corpus from train Corpus
    posCorpus = PlaintextCorpusReader(posTrainCorpus, '.*')
    negCorpus = PlaintextCorpusReader(negTrainCorpus, '.*')

    # Create Plain Text Corpus from test Corpus
    posTest = PlaintextCorpusReader(posTestCorpus, '.*')
    negTest = PlaintextCorpusReader(negTestCorpus, '.*')

    # Create Object of class svm_classifier
    obj1 = svm_classifier(posCorpus,negCorpus)
    
    #feature list for train Corpus
    posFeatList = obj1.featureList(posCorpus)
    negFeatList = obj1.featureList(negCorpus)
    featList = posFeatList + negFeatList

    
    #feature list for test Corpus
    posTestFeatList = obj1.featureList(posTest)
    negTestFeatList = obj1.featureList(negTest)

    #length of positive as well as negative train Corpus
    noPos = len(posCorpus.fileids())
    noNeg = len(negCorpus.fileids())

    #labels array to store 0 (for negative) and 1 (for positive)
    labels = []

    for j in range(noPos):
        labels.append(1)
    for k in range(noNeg):
        labels.append(0)


    
    #Create numpy Array for word frequencies : Feature Vector
    trainFreqArr = np.array(featList)
    trainLabels = np.array(labels)


    #Fit SVM
    
    #L1 and L2 regularization- USE ONLY ONE AT A TIME
    #for L1 regularization
    docClassifier = svm.LinearSVC(loss='l2', penalty='l1', dual=False)

    #for L2 regularization
    docClassifier = svm.LinearSVC()

    docClassifier.fit(trainFreqArr, trainLabels) 

 
    #Create numpy Array for Test Corpus word frequencies : Feature Vector
    posTestarr = np.array(posTestFeatList)
    negTestarr = np.array(negTestFeatList)

    

    # prediction result 
    
    print docClassifier.predict(negTestarr)
    print docClassifier.predict(posTestarr)



# Calling main()
if __name__ == "__main__":
    main()