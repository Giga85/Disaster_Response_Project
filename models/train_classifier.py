# IPython log file

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
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pickle

nltk.download('wordnet') # download for lemmatization
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')


# load data from database
def load_data(database_filepath):
    ''' 
    load_data function loads data from database

    Args:
        database_filepath (string): the path of database file

    Returns:
       datasets: X, Y, category_names
    
    '''
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
    # Reduce words to their stems using Lemmatization
    # words = [PorterStemmer().stem(w) for w in words]
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
    precision = precision_score(test_data, predicted_data, average='micro')
    recall = recall_score(test_data, predicted_data, average='micro')
    
    return {'Accuracy':accuracy, 'f1 score':f1,'Precision':precision, 'Recall':recall}


def build_model():
    ''' 
     build_model creates a model using Pipeline and GridSearchCV

    Args:
        None

    Returns:
        model with pipeline and parameters
    
    '''
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

def evaluate_model(model, X_test, y_test, category_names):
    
    ''' 
     evaluate_model assesses the model 

    Args:
        model (object): The trained model to be evaluated.
        X_test (array-like): The input test data.
        Y_test (array-like): The true labels for the test data.
        category_names (list of str): A list of category names for display.

    Returns:
       test_results_df(data frame): testing results
    
    '''
    
    y_pred_test = model.predict(X_test)
    
    test_results = []
    for i, column in enumerate(y_test.columns):
        result = get_classification_report(y_test.loc[:,column].values, y_pred_test[:,i])
        result['Category'] = column
        test_results.append(result)
    test_results_df = pd.DataFrame(test_results)
    #reorder coumns
    test_results_df = test_results_df.iloc[:,[1, 0, 2, 3, 4]]
    print("Output for Each Category")
    print(test_results_df)

def save_model(model, model_filepath):
    
    ''' 
      save_model stores the model to disk

    Args:
       model: the trained model
       model_filepath (string): the path of model file

    Returns:
       None
       '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        #model = joblib.load("models/cv_t.pkl")

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
