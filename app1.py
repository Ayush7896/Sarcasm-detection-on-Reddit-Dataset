import time
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from better_profanity import profanity
from prettytable import PrettyTable
import datetime
from textblob import TextBlob
import tensorboard
from pyexpat import model
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import keras
from keras.layers import Dropout, BatchNormalization, LSTM, Bidirectional, GlobalMaxPool1D, Input, Activation, Flatten, Embedding, Dense, concatenate, Conv1D, MaxPooling1D
import string
import re
from tqdm import tqdm
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, classification_report
import os
from keras.models import Model
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import warnings
import pickle
# from transformers import TFDistilBertModel
from tensorflow.keras.models import load_model



from yaml import load
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sarcasm Detection On Reddit Dataset", page_icon="")
data = pd.read_csv("train-balanced-sarcasm.csv")
st.title("Sarcasm Detection On Reddit Dataset")


def modeling():
        dictionary_values=pickle.load(open('dictionary_values.pkl','rb'))
        return dictionary_values
dictionary_values=modeling()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
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

def chat(text):
    new_text = []
    for word in text.split():
        if word.upper() in dictionary_values:
            new_text.append(dictionary_values[word.upper()])
        else:
            new_text.append(word)

    done = " ".join(new_text)
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
        new_list = []
        for word in text.split():
            if word in stop_words:
                new_list.append("")
            else:
                new_list.append(word)

        done = list(filter(None, new_list))
        done = " ".join(done)

        return done
    # remove html tags

def remove_html(text):
        return re.sub(r'<.*?>', "", text)

    # removing digits

def remove_numbers(text):
        return re.sub("\d+", "", text)

string1 = string.punctuation
string1 = list(string1)
string1.remove('!')
string1.remove('?')

def remove_punctuation(data):
        for char in string1:
            if char in data:
                data = data.replace(char, " ")
        return data


def preprocess_text(text):
        pre_text = []
        text = chat(text)
        text = decontracted(text)
        text = text.lower()
        text = stopwords1(text)
        text = remove_html(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        pre_text.append(text)
        return pre_text

def profanity_words(text):
        list1 = []
        for sentence in (text):
            profane_word = profanity.contains_profanity(sentence)
            list1.append(profane_word)
        return list1

def sentiment_subjectivity(text):
        list1 = []
        for sentence in (text):
            subjectivity = TextBlob(sentence).sentiment.subjectivity
            list1.append(subjectivity)
        return list1

def sentiment_intensity(text):
        neg_list, pos_list, neutral_list = [], [],[]
        for sentence in (text):
            sentiment_object = SentimentIntensityAnalyzer()
            polarity_scores = sentiment_object.polarity_scores(sentence)
            neg_list.append(polarity_scores['neg'])
            pos_list.append(polarity_scores['pos'])
            neutral_list.append(polarity_scores['neu'])
        return neg_list, pos_list, neutral_list

def count_exclamation(text):
        list1 = []
        for i in text:
            if '!' in i:
                list1.append(1)
            else:
                list1.append(0)

        return list1

def count_question(text):
        list1 = []
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
        
        # sentiment_subj=sentiment_subjectivity(preprocess_list)
        # sentiment_neg,sentiment_pos,sentiment_neu=sentiment_intensity(preprocess_list)
        # exc_mark=count_exclamation(preprocess_list)
        # ques_mark=count_question(preprocess_list)
        # profane_words=profanity_words(preprocess_list)

        # for i in profane_words:
        #     profane_words1=[]
        #     if i==True:
        #         profane_words1.append(1)
        #     else :
        #         profane_words1.append(0)
        
        # sentiment_subj=np.asarray(sentiment_subj)
        # sentiment_neg=np.asarray(sentiment_neg)
        # sentiment_pos=np.asarray(sentiment_pos)
        # sentiment_neu=np.asarray(sentiment_neu)
        # exc_mark=np.asarray(exc_mark)
        # ques_mark=np.asarray(ques_mark)
        # profane_words1=np.asarray(profane_words1)
        final=[padded_sequences]
        model=load_model('model_cnn123.h5')

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



# data_sarcasm=data[data['label']==1]
# data_non_sarcasm=data[data['label']==0]      
# def frequent_top_words(dataframe):
#     top_words=20
#     frequent_words=dataframe.str.cat(sep="")
#     words=nltk.word_tokenize(frequent_words)
#     frequency_disb=nltk.FreqDist(words)
    
#     return frequency_disb.most_common(top_words)
    
# freq_sarcasm=frequent_top_words(data_sarcasm['comment'])
# list1,list2=[],[]
# for i,j in (freq_sarcasm):
#     list1.append(i)
#     list2.append(j)

# plt.figure(figsize=(8,6))
# sns.barplot(y=list1,x=list2)
# plt.title('Top 20 frequent words for sarcastic comments')
# plt.show()

# freq_non_sarcasm=frequent_top_words(data_non_sarcasm['comment'])
# list1,list2=[],[]
# for i,j in (freq_non_sarcasm):
#     list1.append(i)
#     list2.append(j)

# plt.figure(figsize=(8,6))
# sns.barplot(y=list1,x=list2)
# plt.title('Top 20 frequent words for non sarcastic comments')
# plt.show()


# with st.sidebar.header("this is sidebar"):
#         words=""
#         for sentence in data_sarcasm['comment']:
#             tokens=(sentence.split())
#             for i in range(len(tokens)):
#                 tokens[i]=tokens[i].lower()
#             words +=" ".join(tokens)+" "
            
#         wordcloud = WordCloud(width = 800, height = 800,
#                         background_color ='white',
#                         stopwords = stop_words,
#                         min_font_size = 10).generate(words)

#             # plot the WordCloud image                      
#         plt.figure(figsize = (6,6), facecolor = None)
#         plt.imshow(wordcloud)
#         plt.axis("off")
#         plt.tight_layout(pad = 0)
#         plt.title('Sarcastic Comments')
        
#         plt.show()

# with st.sidebar.header("this is sidebar"):
#         words=""
#         for sentence in data_non_sarcasm['comment']:
#             tokens=(sentence.split())
#             for i in range(len(tokens)):
#                 tokens[i]=tokens[i].lower()
#             words +=" ".join(tokens)+" "
            
#         wordcloud = WordCloud(width = 800, height = 800,
#                         background_color ='white',
#                         stopwords = stop_words,
#                         min_font_size = 10).generate(words)

#             # plot the WordCloud image                      
#         plt.figure(figsize = (6,6), facecolor = None)
#         plt.imshow(wordcloud)
#         plt.axis("off")
#         plt.tight_layout(pad = 0)
#         plt.title('Sarcastic Comments')
        
#         plt.show()