# IPython log file

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD


nltk.download('wordnet') # download for lemmatization
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
#[Out]# True
# load data from database
engine = create_engine('sqlite:///data/etl_disaster.db')
df = pd.read_sql_table("message_table",engine)
X = df['message']
Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
def tokenize(text):
    ''' 
     tokenize creates a set of words from text

    Args:
        text (string): list of actual values

    Returns:
        list: a list of wwords
    
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42)

#Train pipeline
pipeline.fit(X_train, y_train)
#[Out]# Pipeline(steps=[('vect',
#[Out]#                  CountVectorizer(tokenizer=<function tokenize_text at 0x000002961F760040>)),
#[Out]#                 ('tfidf', TfidfTransformer()),
#[Out]#                 ('clf',
#[Out]#                  MultiOutputClassifier(estimator=RandomForestClassifier()))])
# predict
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
def get_classification_report(test_data, predicted_data):
    
    '''
    get_classification_report calculates f1 score, precision and recall for each output of the dataset

    Args:
        test_data (list): list of actual data
        predicted_data (list): list of predicted data

    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    '''
    
    accuracy = accuracy_score(test_data, predicted_data)
    f1 = f1_score(test_data, predicted_data,average='micro')
    precision =round( precision_score(test_data, predicted_data, average='micro'))
    recall = recall_score(test_data, predicted_data, average='micro')
    
    return {'Accuracy':accuracy, 'f1 score':f1,'Precision':precision, 'Recall':recall}
#Get the train_results by iterating through the columns using get_classification_report function
def get_results():
    train_results = []
    for i,column in enumerate(y_train.columns):
        result = get_classification_report(y_train.loc[:,column].values,y_pred_train[:,i])
        train_results.append(result)

    #create a dataframe from the train_results
    train_results_df = pd.DataFrame(train_results)
    return train_results_df
#print the results 
train_results_df = get_results()
train_results_df
#[Out]#     Accuracy  f1 score  Precision    Recall
#[Out]# 0   0.998271  0.998271          1  0.998271
#[Out]# 1   0.999237  0.999237          1  0.999237
#[Out]# 2   0.999898  0.999898          1  0.999898
#[Out]# 3   0.998881  0.998881          1  0.998881
#[Out]# 4   0.999593  0.999593          1  0.999593
#[Out]# 5   0.999644  0.999644          1  0.999644
#[Out]# 6   0.999898  0.999898          1  0.999898
#[Out]# 7   0.999797  0.999797          1  0.999797
#[Out]# 8   0.999746  0.999746          1  0.999746
#[Out]# 9   1.000000  1.000000          1  1.000000
#[Out]# 10  0.999949  0.999949          1  0.999949
#[Out]# 11  0.999949  0.999949          1  0.999949
#[Out]# 12  0.999949  0.999949          1  0.999949
#[Out]# 13  0.999949  0.999949          1  0.999949
#[Out]# 14  0.999949  0.999949          1  0.999949
#[Out]# 15  0.999949  0.999949          1  0.999949
#[Out]# 16  0.999847  0.999847          1  0.999847
#[Out]# 17  0.999898  0.999898          1  0.999898
#[Out]# 18  0.999034  0.999034          1  0.999034
#[Out]# 19  0.999644  0.999644          1  0.999644
#[Out]# 20  0.999797  0.999797          1  0.999797
#[Out]# 21  0.999847  0.999847          1  0.999847
#[Out]# 22  1.000000  1.000000          1  1.000000
#[Out]# 23  0.999898  0.999898          1  0.999898
#[Out]# 24  0.999949  0.999949          1  0.999949
#[Out]# 25  0.999949  0.999949          1  0.999949
#[Out]# 26  0.999949  0.999949          1  0.999949
#[Out]# 27  0.999797  0.999797          1  0.999797
#[Out]# 28  0.999491  0.999491          1  0.999491
#[Out]# 29  0.999847  0.999847          1  0.999847
#[Out]# 30  0.999847  0.999847          1  0.999847
#[Out]# 31  1.000000  1.000000          1  1.000000
#[Out]# 32  0.999644  0.999644          1  0.999644
#[Out]# 33  0.999797  0.999797          1  0.999797
#[Out]# 34  0.999847  0.999847          1  0.999847
#[Out]# 35  0.999186  0.999186          1  0.999186
#Check the results
train_results_df.mean()
#[Out]# Accuracy     0.99972
#[Out]# f1 score     0.99972
#[Out]# Precision    1.00000
#[Out]# Recall       0.99972
#[Out]# dtype: float64
parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [10, 20], 
              'clf__estimator__min_samples_split': [2, 4]} 

cv_t = GridSearchCV(pipeline, param_grid=parameters)
cv_t
#[Out]# GridSearchCV(estimator=Pipeline(steps=[('vect',
#[Out]#                                         CountVectorizer(tokenizer=<function tokenize at 0x000002961F760040>)),
#[Out]#                                        ('tfidf', TfidfTransformer()),
#[Out]#                                        ('clf',
#[Out]#                                         MultiOutputClassifier(estimator=RandomForestClassifier()))]),
#[Out]#              param_grid={'clf__estimator__min_samples_split': [2, 4],
#[Out]#                          'clf__estimator__n_estimators': [10, 20],
#[Out]#                          'tfidf__use_idf': (True, False)})
cv_t.fit(X_train, y_train)
#[Out]# GridSearchCV(estimator=Pipeline(steps=[('vect',
#[Out]#                                         CountVectorizer(tokenizer=<function tokenize at 0x000002961F760040>)),
#[Out]#                                        ('tfidf', TfidfTransformer()),
#[Out]#                                        ('clf',
#[Out]#                                         MultiOutputClassifier(estimator=RandomForestClassifier()))]),
#[Out]#              param_grid={'clf__estimator__min_samples_split': [2, 4],
#[Out]#                          'clf__estimator__n_estimators': [10, 20],
#[Out]#                          'tfidf__use_idf': (True, False)})
#print the results 
train_results_df = get_results()
train_results_df
#[Out]#     Accuracy  f1 score  Precision    Recall
#[Out]# 0   0.998271  0.998271          1  0.998271
#[Out]# 1   0.999237  0.999237          1  0.999237
#[Out]# 2   0.999898  0.999898          1  0.999898
#[Out]# 3   0.998881  0.998881          1  0.998881
#[Out]# 4   0.999593  0.999593          1  0.999593
#[Out]# 5   0.999644  0.999644          1  0.999644
#[Out]# 6   0.999898  0.999898          1  0.999898
#[Out]# 7   0.999797  0.999797          1  0.999797
#[Out]# 8   0.999746  0.999746          1  0.999746
#[Out]# 9   1.000000  1.000000          1  1.000000
#[Out]# 10  0.999949  0.999949          1  0.999949
#[Out]# 11  0.999949  0.999949          1  0.999949
#[Out]# 12  0.999949  0.999949          1  0.999949
#[Out]# 13  0.999949  0.999949          1  0.999949
#[Out]# 14  0.999949  0.999949          1  0.999949
#[Out]# 15  0.999949  0.999949          1  0.999949
#[Out]# 16  0.999847  0.999847          1  0.999847
#[Out]# 17  0.999898  0.999898          1  0.999898
#[Out]# 18  0.999034  0.999034          1  0.999034
#[Out]# 19  0.999644  0.999644          1  0.999644
#[Out]# 20  0.999797  0.999797          1  0.999797
#[Out]# 21  0.999847  0.999847          1  0.999847
#[Out]# 22  1.000000  1.000000          1  1.000000
#[Out]# 23  0.999898  0.999898          1  0.999898
#[Out]# 24  0.999949  0.999949          1  0.999949
#[Out]# 25  0.999949  0.999949          1  0.999949
#[Out]# 26  0.999949  0.999949          1  0.999949
#[Out]# 27  0.999797  0.999797          1  0.999797
#[Out]# 28  0.999491  0.999491          1  0.999491
#[Out]# 29  0.999847  0.999847          1  0.999847
#[Out]# 30  0.999847  0.999847          1  0.999847
#[Out]# 31  1.000000  1.000000          1  1.000000
#[Out]# 32  0.999644  0.999644          1  0.999644
#[Out]# 33  0.999797  0.999797          1  0.999797
#[Out]# 34  0.999847  0.999847          1  0.999847
#[Out]# 35  0.999186  0.999186          1  0.999186
#Get the train_results by iterating through the columns using get_classification_report function

train_results = []

for i,column in enumerate(y_train.columns):
    result = get_classification_report(y_train.loc[:,column].values,y_pred_train[:,i])
    train_results.append(result)
    
#create a dataframe from the train_results
train_results_df = pd.DataFrame(train_results)
train_results_df
#[Out]#     Accuracy  f1 score  Precision    Recall
#[Out]# 0   0.998271  0.998271          1  0.998271
#[Out]# 1   0.999237  0.999237          1  0.999237
#[Out]# 2   0.999898  0.999898          1  0.999898
#[Out]# 3   0.998881  0.998881          1  0.998881
#[Out]# 4   0.999593  0.999593          1  0.999593
#[Out]# 5   0.999644  0.999644          1  0.999644
#[Out]# 6   0.999898  0.999898          1  0.999898
#[Out]# 7   0.999797  0.999797          1  0.999797
#[Out]# 8   0.999746  0.999746          1  0.999746
#[Out]# 9   1.000000  1.000000          1  1.000000
#[Out]# 10  0.999949  0.999949          1  0.999949
#[Out]# 11  0.999949  0.999949          1  0.999949
#[Out]# 12  0.999949  0.999949          1  0.999949
#[Out]# 13  0.999949  0.999949          1  0.999949
#[Out]# 14  0.999949  0.999949          1  0.999949
#[Out]# 15  0.999949  0.999949          1  0.999949
#[Out]# 16  0.999847  0.999847          1  0.999847
#[Out]# 17  0.999898  0.999898          1  0.999898
#[Out]# 18  0.999034  0.999034          1  0.999034
#[Out]# 19  0.999644  0.999644          1  0.999644
#[Out]# 20  0.999797  0.999797          1  0.999797
#[Out]# 21  0.999847  0.999847          1  0.999847
#[Out]# 22  1.000000  1.000000          1  1.000000
#[Out]# 23  0.999898  0.999898          1  0.999898
#[Out]# 24  0.999949  0.999949          1  0.999949
#[Out]# 25  0.999949  0.999949          1  0.999949
#[Out]# 26  0.999949  0.999949          1  0.999949
#[Out]# 27  0.999797  0.999797          1  0.999797
#[Out]# 28  0.999491  0.999491          1  0.999491
#[Out]# 29  0.999847  0.999847          1  0.999847
#[Out]# 30  0.999847  0.999847          1  0.999847
#[Out]# 31  1.000000  1.000000          1  1.000000
#[Out]# 32  0.999644  0.999644          1  0.999644
#[Out]# 33  0.999797  0.999797          1  0.999797
#[Out]# 34  0.999847  0.999847          1  0.999847
#[Out]# 35  0.999186  0.999186          1  0.999186
#Improve the pipeline

pipeline_impr = Pipeline([
    ('vect', CountVectorizer()),
    ('best', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
#Train & predict
pipeline_impr.fit(X_train, y_train)
#[Out]# Pipeline(steps=[('vect', CountVectorizer()), ('best', TruncatedSVD()),
#[Out]#                 ('tfidf', TfidfTransformer()),
#[Out]#                 ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))])
#print the results 
train_results_df = get_results()
train_results_df
#[Out]#     Accuracy  f1 score  Precision    Recall
#[Out]# 0   0.998271  0.998271          1  0.998271
#[Out]# 1   0.999237  0.999237          1  0.999237
#[Out]# 2   0.999898  0.999898          1  0.999898
#[Out]# 3   0.998881  0.998881          1  0.998881
#[Out]# 4   0.999593  0.999593          1  0.999593
#[Out]# 5   0.999644  0.999644          1  0.999644
#[Out]# 6   0.999898  0.999898          1  0.999898
#[Out]# 7   0.999797  0.999797          1  0.999797
#[Out]# 8   0.999746  0.999746          1  0.999746
#[Out]# 9   1.000000  1.000000          1  1.000000
#[Out]# 10  0.999949  0.999949          1  0.999949
#[Out]# 11  0.999949  0.999949          1  0.999949
#[Out]# 12  0.999949  0.999949          1  0.999949
#[Out]# 13  0.999949  0.999949          1  0.999949
#[Out]# 14  0.999949  0.999949          1  0.999949
#[Out]# 15  0.999949  0.999949          1  0.999949
#[Out]# 16  0.999847  0.999847          1  0.999847
#[Out]# 17  0.999898  0.999898          1  0.999898
#[Out]# 18  0.999034  0.999034          1  0.999034
#[Out]# 19  0.999644  0.999644          1  0.999644
#[Out]# 20  0.999797  0.999797          1  0.999797
#[Out]# 21  0.999847  0.999847          1  0.999847
#[Out]# 22  1.000000  1.000000          1  1.000000
#[Out]# 23  0.999898  0.999898          1  0.999898
#[Out]# 24  0.999949  0.999949          1  0.999949
#[Out]# 25  0.999949  0.999949          1  0.999949
#[Out]# 26  0.999949  0.999949          1  0.999949
#[Out]# 27  0.999797  0.999797          1  0.999797
#[Out]# 28  0.999491  0.999491          1  0.999491
#[Out]# 29  0.999847  0.999847          1  0.999847
#[Out]# 30  0.999847  0.999847          1  0.999847
#[Out]# 31  1.000000  1.000000          1  1.000000
#[Out]# 32  0.999644  0.999644          1  0.999644
#[Out]# 33  0.999797  0.999797          1  0.999797
#[Out]# 34  0.999847  0.999847          1  0.999847
#[Out]# 35  0.999186  0.999186          1  0.999186
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(cv_t, f)
