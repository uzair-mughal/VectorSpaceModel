import Index
import PreProcessing
import Similarity
from tkinter import *
import tkinter
import json
import os.path
from os import path
from nltk.stem import WordNetLemmatizer

def main_window():

    def process_querry():
        lemmatizer = WordNetLemmatizer()
        stopwords = PreProcessing.read_stopwords()
        query = query_entry.get().lower()
        query = query.split()
        query = [lemmatizer.lemmatize(word) for word in query if word not in stopwords]
        Q_TF = Index.Make_TF_Query(docs,query)
        Q_TF_IDF = Index.Make_TFIDF_Index(Q_TF)
        Q_TF_IDF_Norm = Index.Normalize_Index(Q_TF_IDF)
        COS_SIM = Similarity.Cosine_Similarity(TF_IDF_Norm,Q_TF_IDF_Norm)
        return COS_SIM

    def search_query():
        result_textbar.delete("1.0",tkinter.END)
        docs_entry.delete("1.0",tkinter.END)
        COS_SIM = process_querry()
        ALPHA = alpha_entry.get()
        
        if ALPHA == '':
            ALPHA = 0.005
        else:
            ALPHA = float(ALPHA)
        
        RES = ''
        COUNT = 0
        if CheckVar.get()==1:
            COS_SIM = dict(sorted(COS_SIM.items(), key=lambda item: item[1],reverse=True))

        for key,val in COS_SIM.items():
            if val >=ALPHA:
                RES += str(key)+' '
                COUNT += 1
                
        result_textbar.insert(tkinter.INSERT, RES)
        docs_entry.insert(tkinter.INSERT, str(COUNT))

    window = tkinter.Tk()
    window.title('Information Retrieval (CS317) Assignment-2 18K-0179')
    window.config(bg='PeachPuff2')
    width = 920
    height = 340
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_cod = (screen_width / 2) - (width / 2)
    y_cod = (screen_height / 2) - (height / 2)
    window.geometry('%dx%d+%d+%d' % (width, height, x_cod, y_cod))

    vsm_label = tkinter.Label(window, text='Vector Space Model (VSM)', bg='PeachPuff2', fg='white')
    vsm_label.place(x=360, y=40)

    query_label = tkinter.Label(window, text='Query :', bg='PeachPuff2', fg='white')
    query_label.place(x=75, y=80)

    query_entry = tkinter.Entry(window)
    query_entry.pack()
    query_entry.place(x=160, y=80,height=25 , width=660)

    alpha_label = tkinter.Label(window, text='Alpha (Default : 0.005) :', bg='PeachPuff2', fg='white')
    alpha_label.place(x=75, y=120)

    alpha_entry = tkinter.Entry(window)
    alpha_entry.pack()
    alpha_entry.place(x=240, y=120,height=25 , width=100) 

    search_button = tkinter.Button(window, text="Search",command=search_query, bg='white')
    search_button.place(x=360, y=119, height=27, width= 88)

    CheckVar = IntVar()
    checkbox = Checkbutton(window, text = 'Ranked Order', variable = CheckVar,onvalue = 1, offvalue = 0, height=1, width = 1, bg='white')
    checkbox.pack()
    checkbox.place(x=460, y=120, height=25, width= 120)
   
    
    result_label = tkinter.Label(window, text='Result-Set :', bg='PeachPuff2', fg='white')
    result_label.place(x=75, y=160)

    result_textbar = tkinter.Text(window)
    result_textbar.pack()
    result_textbar.place(x=160, y=160,height=70 , width=660)

    docs_label = tkinter.Label(window, text='Documents Retrieved :', bg='PeachPuff2', fg='white')
    docs_label.place(x=75, y=245)

    docs_entry = tkinter.Text(window)
    docs_entry.pack()
    docs_entry.place(x=240, y=245,height=25 , width=100)
    
    window.mainloop()

if __name__ == "__main__":

    if not path.exists('PreProcessedDocs.json'):
        docs = PreProcessing.filereader()
    else:
        docs = json.load(open('PreProcessedDocs.json'))
        
    if not path.exists('TF_Index.json'):
        TF_Index = Index.Make_TF_Docs(docs)
    else:
        TF_Index = json.load(open('TF_Index.json'))
    
    if not path.exists('TF_IDF.json'):
        TF_IDF = Index.Make_TFIDF_Index(TF_Index)
    else:
        TF_IDF = json.load(open('TF_IDF.json'))
    
    if not path.exists('TF_IDF_Norm.json'):
        TF_IDF_Norm = Index.Normalize_Index(TF_IDF)
    else:
        TF_IDF_Norm = json.load(open('TF_IDF_Norm.json'))



    main_window()

    if not path.exists('PreProcessedDocs.json'):
        j = json.dumps(docs)
        f = open('PreProcessedDocs.json','w')
        f.write(j)
        f.close()
    
    if not path.exists('TF_Index.json'):
        j = json.dumps(TF_Index,indent=2)
        f = open('TF_Index.json','w')
        f.write(j)
        f.close()

    if not path.exists('TF_IDF.json'):
        j = json.dumps(TF_IDF,indent=2)
        f = open('TF_IDF.json','w')
        f.write(j)
        f.close()

    
    if not path.exists('TF_IDF_Norm.json'):
        j = json.dumps(TF_IDF_Norm,indent=2)
        f = open('TF_IDF_Norm.json','w')
        f.write(j)
        f.close()
    