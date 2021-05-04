# Disaster Response Pipeline Project

### Summary of the project

This project is part of the Udacity DataScience Nanodegree.

The main goal is analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the message can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also shows visualizations of the data. 

### How to Run

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files in Repository

* **data** Contains two .csv files and a process.py that when called from terminal creates a sqlite database with some minor data prep
* * **disaster_messages** Contains the id, message that was sent and genre i.e the method (direct, tweet..) the message was sent
* * **disaster_categories** Contains the id and the categories (related, offer, medical assistance..) the message belonged to.
* **process_data.py** ETL that reads, merge and clean the data.
* **ETL Pipeline Preparation.ipynb** jupiter notebook with the process.
* **DisasterResponse.db** sqlite database to store de cleaned data.

### Instructions:





The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository



## Table of Contents
* 1.Installation
* 2.Project Motivation
* 3.File Descriptions
* 4.Licensing, Authors, and Acknowledgements
## 1.Installation
Developed under Python 3.6
Anaconda distribution distribution for Python
* Pandas, Numpy, Scikit-learn, NLP libraries from nltk, Pickle, sqlalchemy

## 2.Project Motivation
T
## 3.File Descriptions

**data** 
contains the twuo .csv files and a process.py which can be called from a terminal "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db". The data folder contains the following files:
* **disaster_messages** Contains the id, message that was sent and genre i.e the method (direct, tweet..) the message was sent
* **disaster_categories** Contains the id and the categories (related, offer, medical assistance..) the message belonged to.
* **process_data.py** ETL that reads, merge and clean the data.
* **ETL Pipeline Preparation.ipynb** jupiter notebook with the process.
* **DisasterResponse.db** sqlite database to store de cleaned data.

**models**
contains train_classifier.py which can be called from terminal  "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl" . The models folder contains the following files:
* **process_data.py** pipeline that trains the NLP model and _creates the **classifier.pkl** in the model folder (because of its size it can't be uploaded)._
* **ETL Pipeline Preparation.ipynb** jupiter notebook with the process.

**app**
contains **run.py** used to deploy the flask app. It can be called from a terminal by running the following command in the **app's directory** "python run.py" (before running make sure to be in the app directory use **cd app** in the terminal.

when the app is running Go to http://0.0.0.0:3001/


## 4 Licensing, Authors, Acknowledgements

Must give credit to Udacity and its partners for the data. You can find the Licensing for the data https://www.udacity.com/course/data-scientist-nanodegree--nd025.