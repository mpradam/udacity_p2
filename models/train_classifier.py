import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import pickle




def load_data(database_filepath):

    '''
    Loads the data from a T_Response table located in a given SQLite database
    
    Input:
    
        database_filepath: Filepath to the SQLite database
    
    Output:
    
        X: Data containing texts
        y: Labels of the categories
        category_names: List of the names of the categories
        
    '''
    
    #Connects to SGlite and fetches table
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('T_Responses', engine)
    
    #Assign values to X, Y and category names
    X = df['message']
    Y = df.iloc[:,4:]
    category_names=Y.columns.values

    return X,Y,category_names

def tokenize(text):
    '''
    This function tokenizes a text after doing some data wrangling

    1. Sets all the text in lower case
    2. Removes punctuation
    3. Tokenizes the text 
    4 Remove stop words
    5. Does lemmatization of the words
    6. Does stemmization of the words
    
    Input:
    
        text: String of text to be tokenized
    
    Output:
    
        words: List of tokenizes,lemmatized and stemmed words.
    
    '''
     #Lower case
    text = text.lower()
    #Removes puntuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    #Tokenizes
    words = word_tokenize(text)
    #Removes stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    #Lemmatizes
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    #Stemmizes
    words= [PorterStemmer().stem(w) for w in words]

    return words


def build_model():
    '''
    Builds the pipeline of the model to be implemented
    
    Input:
    
        No input.
    
    Output:
    
        cv: Grid search of pipeline
    '''
    #Build pipeline of the model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    #Grid of parameters
    parameters = {
        'clf__estimator__max_depth': [ 80, 90, 100],
        'clf__estimator__n_estimators': [5, 10, 15]}

    #Grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model on test data
    
    Input:
    
    model: Fitted model to be evaluated
        X_test: Test features
        Y_test: Test labels
        category_names: Names of the catrgories
        
    Output:
        Print model statistics for each category
    
    '''
    #Calculates predictions
    y_prediction_test = model.predict(X_test)
    
    #Prints model performance
    print(classification_report(Y_test.values, y_prediction_test, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save the model as pickle file
    Input:
        
        model: Trained model ready to be deployed
        model_filepath: Path and file name to stor pickle   
    
    Return:
        NA
    '''
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    
    
    


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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()