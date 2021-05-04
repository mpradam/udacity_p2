import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Imports data form cvs filepaths and merges them into a single dataframe
    
    Input:
    
        messages_filepath: Filepath containing tha messages cvs
        categories_filepath: Filepath containing tha categories cvs
        
    Output:
    
        df: pandas dataframe that contains a merged of the two cvs
    """
    
    #loads data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merges datasets
    df = messages.merge(categories, on="id")
    
    return df
    

def clean_data(df):
    """
    Takes a dataframe and cleans it for the ML pipeline     
    
    Input:
    
        df: a pandas datagrame of "dirty" data of messages and categories 
    
    Output:
    
        df: Clean pandas dataframe cotaining data for ml purposes
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # 2 will be converted to 0
    categories['related'] = categories['related'].replace(2, 0)
    
    # The child alone column is nos used
    categories.drop("child_alone", axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df=df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned data in SQlite database.
    
    Input:
        df: pandas dataframe with the cleaned data of messages and categories.
        database_filename: Name of the output database
    
    Output:
        No output.
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('T_Responses', engine, index=False, if_exists='replace')
    
    pass  


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