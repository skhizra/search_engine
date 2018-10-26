
'''
   utility functions for processing terms

    shared by both indexing and query processing
'''

from nltk.stem import PorterStemmer


def isStopWord(word):
    ''' using the NLTK functions, return true/false'''
    
    #open the stopwords file    
    file = open("stopwords", "r") 
    #read line by line and create a list
    words = [line.rstrip('\n') for line in file]
    #return true if stopword else false
    for w in words:
        if word == w:
            return True
    
    return False

    
    


def stemming(word):
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''
    #stemming
    ps = PorterStemmer()
    return ps.stem(word)

    


