import math

DocIDs = list(range(1,51))

def Cosine_Similarity(TF_IDF,Q_TF_IDF):
    Cos_Sim = dict.fromkeys(DocIDs, 0)
    for key in Q_TF_IDF.keys():
        for i in range (1,51):
            Cos_Sim[i] += TF_IDF[key][str(i)] * Q_TF_IDF[key][str(i)] 

    return Cos_Sim