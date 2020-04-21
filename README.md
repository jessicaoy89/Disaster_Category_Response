# Disaster_Category_Response
Identification of disaster messages and response categories using ETL, NLP and ML pipelines

## Table of Contents:
[Installation](##Installation)
[Project Motivation](##Project Motivation)
[Data Descriptions](##Data Descriptions)
[File Descriptions](##File Descriptions)
[Results](##Results)
[Licensing, Authors, Acknowledgements](##Licensing, Authors, Acknowledgements)

## Installation
There is no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.

To run the project in its root directory, proceed with the following steps in command window: 
1. To run ETL pipeline to clean data and store it in a database, input:
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run ML pipeline to train and save the proper classifier, input:
  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. To view the results in a web app, go to the app's directory in command, and input:
  `python run.py`
  Go to http://localhost:3001 to view results.

## Project Motivation
This project aims to analyze disaster messages and determine their categories, according to which helps will decide on conducting the proper response activities.

## Data Descriptions
There is two .csv files available in the /data folder to perform the pipelines:
- disaster_messages.csv: messages sent during disasters via social media or directly to disaster response organizations
- disaster_categories.csv: category of disaster messages (with a total of 36 options)

## File Descriptions
There is three .py files available to showcase work related to the above goals:
- data/process_data.py: ETL pipeline to clean the data from disaster_messages.csv and disaster_categories.csv and store it in a sqlite database named DisasterResponse.db
- models/train_classifier.py: ML pipeline to train and classify the disaster_messages into 36 categories, then store the trained model as a classifier.pkl file
- apps/run.py: a web app that visualizes the classification results in a webpage.

There is a note file that shows the classification report with test data.

## Results
In this project, I tested both Decision Tree and Random Forest Classifiers and decided to use the RandomForestClassifier for categorization. According to the classification report, the project showed good precision and high scores especially for cagetories with large sample sizes (e.g. with precision > 0.9).

In order to further improve the model, using larger sample sizes and tuning more combinations of parameters with GridSearchCV are desired.

## Licensing, Authors, Acknowledgements
Credit goes to [Figure Eight](https://appen.com/) for the data. The code is under open source GNU, feel free to use the code here as you would like!
