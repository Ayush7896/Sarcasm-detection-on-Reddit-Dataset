from pyexpat import model
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import keras
from keras.layers import Dropout,BatchNormalization,LSTM,Bidirectional,GlobalMaxPool1D,Input,Activation,Flatten,Embedding,Dense,concatenate,Conv1D,MaxPooling1D
import string
import re
from tqdm import tqdm
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,classification_report
import os
from keras.models import Model
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import warnings

from yaml import load
warnings.filterwarnings('ignore')
import tensorboard
from textblob import TextBlob
import os
import tensorboard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
from keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from prettytable import PrettyTable
from better_profanity import profanity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
import time

st.set_page_config(page_title="Sarcasm Detection On Reddit Dataset",page_icon="")
data=pd.read_csv("train-balanced-sarcasm.csv")

def main():
    st.title("Sarcasm Detection On Reddit Dataset")
    @st.cache
    def modeling():
        model=load_model('model.h5')
        return model

 
    strings='''AFAIK=As Far As I Know
    AFK=Away From Keyboard
    ASAP=As Soon As Possible
    ATK=At The Keyboard
    ATM=At The Moment
    A3=Anytime, Anywhere, Anyplace
    BAK=Back At Keyboard
    BBL=Be Back Later
    BBS=Be Back Soon
    BFN=Bye For Now
    B4N=Bye For Now
    BRB=Be Right Back
    BRT=Be Right There
    BTW=By The Way
    B4=Before
    B4N=Bye For Now
    CU=See You
    CUL8R=See You Later
    CYA=See You
    FAQ=Frequently Asked Questions
    FC=Fingers Crossed
    FWIW=For What It's Worth
    FYI=For Your Information
    GAL=Get A Life
    GG=Good Game
    GN=Good Night
    GMTA=Great Minds Think Alike
    GR8=Great!
    G9=Genius
    IC=I See
    ICQ=I Seek you (also a chat program)
    ILU=ILU: I Love You
    IMHO=In My Honest/Humble Opinion
    IMO=In My Opinion
    IOW=In Other Words
    IRL=In Real Life
    KISS=Keep It Simple, Stupid
    LDR=Long Distance Relationship
    LMAO=Laugh My A.. Off
    LOL=Laughing Out Loud
    LTNS=Long Time No See
    L8R=Later
    MTE=My Thoughts Exactly
    M8=Mate
    NRN=No Reply Necessary
    OIC=Oh I See
    PITA=Pain In The A..
    PRT=Party
    PRW=Parents Are Watching
    QPSA?=Que Pasa?
    ROFL=Rolling On The Floor Laughing
    ROFLOL=Rolling On The Floor Laughing Out Loud
    ROTFLMAO=Rolling On The Floor Laughing My A.. Off
    SK8=Skate
    STATS=Your sex and age
    ASL=Age, Sex, Location
    THX=Thank You
    TTFN=Ta-Ta For Now!
    TTYL=Talk To You Later
    U=You
    U2=You Too
    U4E=Yours For Ever
    WB=Welcome Back
    WTF=What The F...
    WTG=Way To Go!
    WUF=Where Are You From?
    W8=Wait...
    7K=Sick:-D Laugher'''
    x1=strings.split("\n")
    dict1={}
    for i in x1:
        x2=(i.split("="))
        dict1[x2[0]]=x2[1]
    stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", 
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", 
            "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
            "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
            "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", 
            "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such" "only", "own", "same", "so", "than", "too", "very", 
            "s", "t", "can", "will", "just", "don", "should", "now"]
    # path_to_glove_file = r'C:\Users\DELL\Desktop\Sarcasm\glove.6B.100d.txt'
    # embeddings_index = {}
    # with open(path_to_glove_file,encoding='utf-8') as f:
    #     for line in f:
    #         word, coefs = line.split(maxsplit=1)
    #         coefs = np.fromstring(coefs, "f", sep=" ")
    #         embeddings_index[word] = coefs
    def chat(text):
        new_text=[]
        for word in text.split():
            if word.upper() in dict1:
                new_text.append(dict1[word.upper()])
            else:
                new_text.append(word)
                
        done=" ".join(new_text)
        return done
    def decontracted(phrase):
        
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
    
        return phrase


    def stopwords1(text):
        new_list=[]
        for word in text.split():
            if word in stop_words:
                new_list.append("")
            else:
                new_list.append(word)

        done=list(filter(None,new_list))
        done=" ".join(done)
        
        return done
    # remove html tags
    def remove_html(text):
        return re.sub(r'<.*?>',"",text)


    # removing digits
    def remove_numbers(text):
        return re.sub("\d+", "", text)

    string1=string.punctuation
    string1=list(string1)
    string1.remove('!')
    string1.remove('?')

    def remove_punctuation(data):
        for char in string1:
            if char in data:
                data=data.replace(char," ")
        return data

    def preprocess_text(text):
        pre_text=[]
        text=chat(text)
        text=decontracted(text)
        text=text.lower()
        text=stopwords1(text)
        text=remove_html(text)
        text=remove_numbers(text)
        text=remove_punctuation(text)
        pre_text.append(text)
        return pre_text

    def profanity_words(text):
        list1=[]
        for sentence in (text):
            profane_word=profanity.contains_profanity(sentence)
            list1.append(profane_word)   
        return list1
    def sentiment_subjectivity(text):
        list1=[]
        for sentence in (text):
            subjectivity=TextBlob(sentence).sentiment.subjectivity
            list1.append(subjectivity)
        return list1
    def sentiment_intensity(text):
        neg_list,pos_list,neutral_list=[],[],[]
        for sentence in (text):
            sentiment_object= SentimentIntensityAnalyzer()
            polarity_scores=sentiment_object.polarity_scores(sentence)
            neg_list.append(polarity_scores['neg'])
            pos_list.append(polarity_scores['pos'])
            neutral_list.append(polarity_scores['neu'])  
        return neg_list,pos_list,neutral_list

    def count_exclamation(text):
        list1=[]
        for i in text:
            if '!' in i:
                list1.append(1)
            else:
                list1.append(0)
                
        return list1
    
    def count_question(text):
        list1=[]
        for i in text:
            if '?' in i:
                list1.append(1)
            else:
                list1.append(0)       
        return list1
  

    def main_function(text):
        preprocess_list=preprocess_text(text)
        tokenizer=Tokenizer(num_words=40000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',oov_token='<OOV>')
        tokenizer.fit_on_texts(preprocess_list)
        x=tokenizer.texts_to_sequences(preprocess_list)
        maxlen = 100
        word_index=tokenizer.word_index

        padded_sequences=pad_sequences(x,maxlen=maxlen,padding='post',truncating='post')
        
        sentiment_subj=sentiment_subjectivity(preprocess_list)
        sentiment_neg,sentiment_pos,sentiment_neu=sentiment_intensity(preprocess_list)
        exc_mark=count_exclamation(preprocess_list)
        ques_mark=count_question(preprocess_list)
        profane_words=profanity_words(preprocess_list)

        for i in profane_words:
            profane_words1=[]
            if i==True:
                profane_words1.append(1)
            else :
                profane_words1.append(0)
        
        sentiment_subj=np.asarray(sentiment_subj)
        sentiment_neg=np.asarray(sentiment_neg)
        sentiment_pos=np.asarray(sentiment_pos)
        sentiment_neu=np.asarray(sentiment_neu)
        exc_mark=np.asarray(exc_mark)
        ques_mark=np.asarray(ques_mark)
        profane_words1=np.asarray(profane_words1)
        final=[padded_sequences,sentiment_subj.reshape(-1,1),sentiment_neg.reshape(-1,1),sentiment_pos.reshape(-1,1),sentiment_neu.reshape(-1,1),exc_mark.reshape(-1,1),ques_mark.reshape(-1,1),profane_words1.reshape(-1,1)]
        model=load_model('model.h5')
        pred=model.predict(final)
        final_predictions=[]
        if pred[0]>0.5:
            final_predictions.append(1)
        else:
            final_predictions.append(0)
            
        if 1 in final_predictions:
            return ("The above comment is sarcastic")
        else:
            return ("The above comment is non sarcastic")

    text=st.text_area("Enter your comment","Type Here")
    if st.button("Click to Predict"):
        with st.spinner('Wait for it...'):
            time.sleep(2)
        prediction=main_function(text)
        
        st.caption("Predicted Value:")
        st.write(prediction)

if __name__=='__main__':
    main()
