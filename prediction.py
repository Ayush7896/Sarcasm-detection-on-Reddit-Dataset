from pyexpat import model
from statistics import mode
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import keras

import string
import re
from tqdm import tqdm
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
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
from textblob import TextBlob
import os
import datetime
from prettytable import PrettyTable
from better_profanity import profanity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
import time
from app1 import main_function
import pickle
# data=pd.read_csv("train-balanced-sarcasm.csv")

def main():
    # st.title("Sarcasm Detection On Reddit Dataset")
    


    text=st.text_area("Enter your comment","Type Here")
    if st.button("Click to Predict"):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        prediction=main_function(text)
        
        st.caption("Predicted Value:")
        st.write(prediction)

if __name__=='__main__':
    main()
