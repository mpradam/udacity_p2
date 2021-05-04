# Disaster Response Pipeline Project

## Table of Contents
1. Summary of the project
2. How to run
3. File Descriptions


### 1. Summary of the project

This project is part of the Udacity DataScience Nanodegree.

The main goal is analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the message can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also shows visualizations of the data. 

### 2. How to Run

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### 3. File Descriptions

* **app** Cointains the flask app code and q "tun.py" file where the visualizations where built.

* **data** Contains two .csv files and a python file
    * **disaster_messages.csv** Dataset containing the text of the messages.
    * **disaster_categories.csv** Data set containing lables of the text messages.
    * **process_data.py** Python file trat does some minor data prep and creates a sqlite database

* **models** Contains a python file that has the following steps
    * **Data load** Loads the data from the previously created sqlite database
    * **Pipeline** Creates a pipeline with a costume built tokenizer, TDIDF and a Multiclass clasifires
    * **Train** Trains the model on a gridsearch and selects the best performer
    * **Validate** Validates accuracy, precision, recall and f1_score.
    * **Export** Export the model as a Pickle.

* **notebooks**
    * **ETL Pipeline Preparation.ipynb** jupiter notebook with the etl procces.
    * **ML Pipeline Preparation.ipynb** jupiter notebook with the ml procces.
    


