#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar  1 20:41:37 2018

@author: shufakhizra
"""

'''

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

'''
import util
import cran
from collections import OrderedDict
import math
import pickle
import datetime
from nltk.tokenize import RegexpTokenizer
import re
import sys


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        '''sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
        #ToDo
        return len(self.positions)


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        self.sorted_postings= [] # may sort them by docID for easier query processing
        self.sortedp = OrderedDict()   #stores the sorted posting list                   

    def add(self, docid, pos):
        ''' add a posting'''
        if docid not in self.posting:
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)
        
        
    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
        #ToDo
        self.sortedp = OrderedDict(sorted(self.posting.items()))
        
        
        


class InvertedIndex:

    def __init__(self):
        self.items = {} # list of IndexItems
        self.nDocs = 0  # the number of indexed documents
        
        self.df = OrderedDict() #document frequency
        self.index = OrderedDict() #sorted index by terms
        self.idfs = {}   #idfs of each term
        self.tf = {}     # weights of each term in documents
        self.dictionary = {}   #store the terms of each document as a list: used to compute the sum of squares.
        self.myDicts = []      #to load the index as list after writing to disk
        self.N = totalDocuments()  #total documents 
        

    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        #ToDo: return the IDF of the term
        
        #compute idf
        if self.df[term] !=0: 
            return math.log10(self.N/self.df[term])
        else: 
            return 0



    def indexDoc(self, docs): # indexing a Document object
    
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''

        # ToDo: indexing only title and body; use some functions defined in util.py
        # (1) convert to lower cases,
        # (2) remove stopwords,
        # (3) stemming
        
        #lower case title and body
        t = docs.title.lower()
        b = docs.body.lower()
        
        self.nDocs = self.nDocs + 1

        #remove numbers
        t = re.sub(r'\d+', '', t)
        b = re.sub(r'\d+', '', b)

        #tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(t)

        #stopword removal
        words = []        
        for i in tokens:
            if util.isStopWord(i) == False:
                words.append(i)
        
        tokens = tokenizer.tokenize(b)
        for i in tokens:
            if util.isStopWord(i) == False:
                words.append(i)
        #stemming
        word = []
        for i in words:
            word.append(util.stemming(i))

        #to store the terms of each document as list to compute unit vector of documents
        self.dictionary[docs.docID] = []

        #add each term to the index
        for pos,i in enumerate(word):                        
            if i not in self.items:
                self.items[i] = IndexItem(i)  # create an IndexItem object for each term
            self.items[i].add(docs.docID, pos+1)  # add documents and positions to posting dictionary
            self.dictionary[docs.docID].append(i) #adds each term to the list appearing in a document
            
        self.dictionary[docs.docID] = set(self.dictionary[docs.docID])    #removes duplicate terms from the list
  
        
        #computing tf-idf                
        if self.nDocs == self.N:                       
    
            #sort the index by terms
            self.index = OrderedDict(sorted(self.items.items()))

            #compute document frequency for each term
            for term in self.index:
                self.df[term] = len(self.index[term].posting)
                self.index[term].sort()    #sort the posting by docID
        
            #compute the term frequency for each term in a document
            for term in self.index:
                self.tf[term] = {}
                for docc in self.index[term].sortedp:
                    if len(self.index[term].sortedp[docc].positions) == 0:
                        self.tf[term][docc] = 0
                    else:
                        #compute  (1+ log(tf))*idf 
                        self.tf[term][docc] = (1 + math.log10(len(self.index[term].sortedp[docc].positions))) * self.idf(term)

            
            #compute the sum of squares of each term in the document and calculate square root 
            sums = {}
            for d in self.dictionary:
                sums[d] = 0
                for i in self.dictionary[d]:
                    sums[d] += math.pow(self.tf[i][d],2)
                sums[d] = math.sqrt(sums[d]) 
                
                
            #Divide tf*idf/ sqrt(sum of sqaures) and store in tf dictionary -> weights for each term in the documents
            for term in self.index:
                self.idfs[term] = self.idf(term)  #compute the idf of each term in the index and store in dictionary
                for docc in self.index[term].sortedp:
                    if sums[docc] == 0:
                        self.tf[term][docc] = 0
                    else:           
                        self.tf[term][docc] = self.tf[term][docc]/sums[docc]
                
        
         
         
    def sort(self):
        ''' sort all posting lists by docID'''
        #ToDo
        #sort the index by terms
        self.index = OrderedDict(sorted(self.items.items()))

        

    def find(self, term):
        return self.items[term]

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        
        #saving the index as objects using pickle
        fileObject = open(filename,'wb') 
        MyDicts = [self.index, self.tf, self.idfs]
        pickle.dump(MyDicts,fileObject)   
        fileObject.close()
       
    def load(self, filename):
        ''' load from disk'''
        # ToDo
    
        #load the index
        self.myDicts = pickle.load( open (filename, "rb") )
        


def test(filename):
    ''' test your code thoroughly. put the testing cases here'''
    
    
    #print the inverted index
    obj = InvertedIndex()
    obj.load(filename) #call load function to load the index

    #print length of inverted index
    print("number of terms in index are: ", len(obj.myDicts[0]))

    # print the index
    print("index")
    for i in obj.myDicts[0]:
            #print term
            print(i)
            for j in obj.myDicts[0][i].sortedp:
                #print docid
                print(j, " " , end='')
                for k in obj.myDicts[0][i].sortedp[j].positions:
                    #print positions of term in doc
                    print(k, " ", end='')
                print("\n")    
            print("\n")         

    '''
    #print weights of each term in document        
    print("weights")    
    for term in obj.myDicts[1]:
        for docs in obj.myDicts[1][term]:
            print(term," ", docs, " ", obj.myDicts[1][term][docs])

    #print idf of each term                
    print("idf")            
    for term in obj.myDicts[2]:
        print(term, " ", obj.myDicts[2][term])
    '''                        
   

def indexingCranfield():
    #ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file"
    # the index is saved to index_file
    
    #read arguements from command line
    file = sys.argv[1]
    filename = sys.argv[2]
    
    filename += ".p"
    
    #create obejct 
    i = InvertedIndex()
    
    cf = cran.CranFile (file)
    
    print(datetime.datetime.now())
    for docs in cf.docs:
        i.indexDoc(docs)  #call indexDoc to create index for each doc

    #save the index to disk
    i.save(filename)

    print(datetime.datetime.now())

    
    
def totalDocuments():
    #total number of documents
    cf = cran.CranFile (sys.argv[1])
    return len(cf.docs)
    
    

if __name__ == '__main__':
    #test()
    indexingCranfield()
    test("index_file.p")
