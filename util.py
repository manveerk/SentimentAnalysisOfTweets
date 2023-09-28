import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report,balanced_accuracy_score, precision_recall_curve

def preprocess_tweet(df, col):
    """
        Remove handles, Retweet/ Quote tweet codes, URLs, white spaces, emojis, # symbol in hashtags
        Convert to lower case
    """
    df[col] = df[col].apply(lambda x: re.sub(r'@[\S]+', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'(\bRT\b|\bQT\b)', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'http[\S]+', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x)))
    df[col] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x)))
    

def tokenize(df, col):
    """
        Converts string in the column to tokens
    """
    text = ' '.join(df[col].to_list())
    tokens = nltk.word_tokenize(text)
    return tokens

stop_words = set(stopwords.words('english'))
def no_stopwords(text):
    """
        Removes stopwords from text
    """
    lst = [word for word in text if word not in stop_words]
    return lst

def term_frequency(df):
    tf1 = (df['tweet'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index())
    tf1.columns = ['words', 'tf']
    tf1 = tf1.sort_values(by='tf', ascending=False).reset_index()
    return tf1

def stemming(token_list):
    ss = PorterStemmer()
    lst = [ss.stem(w) for w in token_list]
    return lst

def lemmatization(df):
    df['lem'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df['lem'].head()

def get_metrics(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    """
        Function to get training and validation F1, recall, precision scores
        Instantiate model and pass the model into function
        Pass X_train, y_train, X_val, Y_val datasets
        Pass in calculated model.predict(X) for y_pred
    """    
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    #aps_tr = aps(X_tr, y_tr, model)
    #aps_val = aps(X_val, y_val, model)
    
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    #print('Training Average Precision Score: ', aps_tr)
    #print('Validation Average Precision Score: ', aps_val)