# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' 
    load_data function loads data from files and merge them together

    Args:
        messages_filepath (string): the path of messages file
        categories_filepath (string): the path of categories file

    Returns:
       data frame: df
       '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    #2. Merge datasets.
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    ''' 
    clean_data transforms data from data frame 

    Args:
        df (data frame): loaded data frame

    Returns:
       data frame: clean data frame
       
       '''
    # 3. Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0,]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =row.apply(lambda x:x.split("-")[0] )
    # rename the columns of `categories`
    categories.columns =category_colnames
    #categories.head()
    
    # 4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =categories[column].apply(lambda x:x.split("-")[-1])
            
        # convert column from string to numeric
        categories[column] =categories[column].astype('int')
    
    #5. Replace categories column in df with new category columns.
      
    # drop the original categories column from `df`
    df.drop('categories',inplace=True, axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis = 1)
    
    #6. Remove duplicates.
    df = df.drop_duplicates()
      # drop the rows that contains valued 2 in related column from `categories`
    #categories = categories[categories['related'] < 2]
    indexRelated = df[df['related'] > 1].index
    df.drop(indexRelated, inplace=True)
    return df

#7. Save the clean dataset into an sqlite database.
def save_data(df, database_filename):
    ''' 
    save_data stores data frame to a database file

    Args:
       df (data frame): the clean data frame
        database_filename (string): the path of database file

    Returns:
       None
       '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_table', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()