import json
import os.path
from os import path
import math
import copy

DocIDs = []
for i in range(1,51):
    DocIDs.append(str(i))
    
# this function make the Positional TF_Index for all the docs
def Make_TF_Docs(docs):
    TF_Index = {} 
    for i in range(len(docs)):
        for word in docs[i]:
            if word not in TF_Index.keys():
                TF_Index[word] = dict.fromkeys(DocIDs, 0)

            TF_Index[word][str(i+1)] += 1
            
    TF_Index = dict(sorted(TF_Index.items(), key=lambda item: item[0]))  
    return TF_Index

def Make_TF_Query(docs,query):
    TF_Index = {}
    for i in range(len(docs)):
        for word in docs[i]:
            if word in query:
                if word not in TF_Index.keys():
                    TF_Index[word] = dict.fromkeys(DocIDs, 0)

                TF_Index[word][str(i+1)] += 1
            
    TF_Index = dict(sorted(TF_Index.items(), key=lambda item: item[0])) 
    return TF_Index

def Make_TFIDF_Index(TF_Index):
    IDF_Index = copy.deepcopy(TF_Index)
    for key in IDF_Index.keys():
        df = 0
        for i in range(1,51):
            if IDF_Index[key][str(i)]>0:
                df += 1
        
        idf = math.log10(df)/50
        for i in range(1,51):
                IDF_Index[key][str(i)] *= idf
    return IDF_Index

def Normalize_Index(TF_IDF):
    TF_IDF_Norm = copy.deepcopy(TF_IDF)
    for i in range(1,51):
        den = 0
        for key in TF_IDF_Norm.keys():
            den += TF_IDF_Norm[key][str(i)]**2

        den = math.sqrt(den)

        for key in TF_IDF_Norm.keys():
            try:
                TF_IDF_Norm[key][str(i)] /= den
            except:
                TF_IDF_Norm[key][str(i)] = 0
    return TF_IDF_Norm

