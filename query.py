#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar  1 23:23:28 2018

@author: shufakhizra
"""


'''
query processing

'''
#import index as index
import pickle
import cranqry
import util
import norvig_spell
import math
from collections import OrderedDict
import operator
from itertools import islice
from functools import reduce
import sys
import re
from nltk.tokenize import RegexpTokenizer

class QueryProcessor:

    def __init__(self, query, index):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index = index
        self.words = []
        #self.docs = collection
        self.myDicts = []
        self.docs = []

    def preprocessing(self):        
        ''' apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector. Note that
            spelling corrector should be applied before stopword
            removal and stemming (why?)'''

        #ToDo: return a list of terms
        
        #lower-case query
        self.raw_query = self.raw_query.lower()
        
        #eliminate numbers
        self.raw_query= re.sub(r'\d+', '', self.raw_query)

        #tokenizing
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(self.raw_query)

        self.words = []
        
        #spell correction, stop word removal, stemming
        for i in tokens:
            i = norvig_spell.correction(i)
            if util.isStopWord(i) == False:
                self.words.append(util.stemming(i))
        
       
    def booleanQuery(self):
        ''' boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C" for retrieving posting lists and merge them'''
        #ToDo: return a list of docIDs
        
        #loading index
        self.myDicts = pickle.load( open (self.index, "rb") )

        
        #collecting docs that contain the term in query        
        qdoc = {}
        for terms in set(self.words):
            qdoc[terms] = []
            if terms in self.myDicts[0]: #check if term in index

                for docs in self.myDicts[0][terms].sortedp:
                    qdoc[terms].append(docs)
        
        
        #sort the lists in ascending oder for efficient merging, return the list of keys in asceding order
        list1 = sorted(qdoc, key=lambda k: len(qdoc[k]), reverse= False)
        
        #create ordered dictionary
        ddoc = OrderedDict()
        
        #stores the lists in ascending order in dictionary
        for i in list1:
            ddoc[i] = qdoc[i]
        
        #list of documents to merge in ascending order
        m = list(ddoc.values())
        
        #And operation on the list of documents
        x = list(reduce(lambda x,y: set(x)&set(y),m))
        
        
        #print results
        print("----Boolean Ranking---")
        if len(x) == 0:
            print("No matching documents")
        else: 
            for i in x:
                print(i)
            
      
        


    def vectorQuery(self, k):
        ''' vector query processing, using the cosine similarity. '''
        #ToDo: return top k pairs of (docID, similarity), ranked by their cosine similarity with the query in the descending order
        # You can use term frequency or TFIDF to construct the vectors
        
        

        #compute the term frequency for terms in query
        queryf = {}       
        for i in set(self.words):
            count = 0
            for j in self.words:
                if j == i:
                    count +=  1
            queryf[i] = count
        
        #loading index
        self.myDicts = pickle.load( open (self.index, "rb") )
        
        #creating query vector with length equal to number of terms in index (bag of words)
        qvec = [0]*len(self.myDicts[0])
        
                 
        #create query vector with the weights of each term 
        pos =0
        for term in self.myDicts[0]:
            if term in queryf:
                if queryf[term] != 0:
                    if term in self.myDicts[2]:
                        qvec[pos] = (1 + math.log10(queryf[term])) * self.myDicts[2][term]  # compute tf* idf
            pos += 1    
        
        #sqaure root of sum of weights
        norm = math.pow(sum(i*i for i in qvec), 0.5)
        
        #vector of tf*idf weigths, divide by norm
        for pos, x in enumerate(qvec):
            if norm!=0:
                qvec[pos] = x/norm
            else: qvec[pos] = 0
        
        #list of documents in the index
        self.docs = list(self.myDicts[0].keys())


        #collecting docs that contain the term in query        
        qdoc = []
        for terms in set(self.words):
            if terms in self.myDicts[0]:
                for docs in self.myDicts[0][terms].sortedp:
                    qdoc.append(docs)
       
        #eliminate duplicate docids
        qdocs = set(qdoc)
        
        #computing the vector of weights for all the documents 
        dvec = {}
        for doc in qdocs:
            dvec[doc] = [0]* len(self.myDicts[0]) #create list of length equal to number of terms in index
            for pos,term in enumerate(self.myDicts[0]): 
                if doc in self.myDicts[1][term]:
                    dvec[doc][pos] = self.myDicts[1][term][doc]  #storing the weights in a vector for each document
        
        #create ordered dictionary
        result = OrderedDict()
        
        #computing cosine similiarity of query and each document
        for doc in dvec:
            result[doc]= sum([a*b for a,b in zip(qvec,dvec[doc])])
        
        #ranking in descending order
        ranking = OrderedDict(sorted(result.items(), key=operator.itemgetter(1), reverse = True))
        
        print("-----Vector Results-------")
        
        #selecting only the top # of documents 
        order = list(islice(ranking.items(), k))    
        
        #print the tuples
        for i, j in order:
            if j!=0:
                print("(", i, ",",  j , ")")
            

def test():
    ''' test your code thoroughly. put the testing cases here'''
    #print 'Pass'

def query():
    ''' the main query processing program, using QueryProcessor'''
    
    #i = QueryProcessor() 
    
    # ToDo: the commandline usage: "echo query_string | python query.py index_file processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents

    #reading command line arguments
    index = sys.argv[1]
    algo = sys.argv[2]
    file = sys.argv[3]
    qid = sys.argv[4]
    
    index += ".p"

    #number of results to display
    k = 6

    #loading the query.text
    qrys =  cranqry.loadCranQry(file)
    obj = QueryProcessor(qrys[qid].text,index) 
    obj.preprocessing() 

    
    if algo == '0':
        obj.booleanQuery()
    if algo == '1':
        
        obj.vectorQuery(k)




if __name__ == '__main__':
    #test()
    query()
   