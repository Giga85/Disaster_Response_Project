# import libraries
import sys
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


# load data from database
def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("message_table", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names

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


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [10, 20], 
              'clf__estimator__min_samples_split': [2, 4]}
    
    cv_t = GridSearchCV(pipeline, param_grid=parameters)

    return cv_t

def save_model(model, model_filepath):
    model = build_model()
    joblib.dump(model, model_filepath)

    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
