# Disaster_Response_Project
Disaster_Response_Project
### 1. Introduction:

The goal of Disaster_Response_Project is to help emergency responders identify and classify the relevant messages during a disaster

In this project, I classify disaster messages from provided datasets. The project includes the following tasks:

1. Build ETL Pipeline to read datasets, transform and finally save the clean data to a database
2. Create Machine learning pipeline to train, build model and save the model to a pkl file
3. Build an wweb app to classify the disaster messages using the built model.

### 2. Structure of the project:

- Workspace
  
      - app
  
          - templates - consists of web pages
  
          - run.py - python script used to run the app
  
      - data
  
          - disaster_messages.csv
  
          - disaster_categories.csv
  
          - etl_disaster.db
  
      - models
  
          - train_classifier.py
  
          - cv_t.pkl
  
      - Readme
  
      - ETL Pipeline Preparation.ipynb
  
      - ETL Pipeline Preparation.html
  
      - ML Pipeline Preparation.ipynb
  
      - ML Pipeline Preparation.html
  
### 3. Instructions running the project:

1. Create an ETL pipeline that cleans data and stores in database:
    - To run ETL pipeline: Run the following commands in the project's root directory:

       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/etl_disaster.db`
2. Create a ML pipeline that trains classifier and saves the model to to cv_t.pkl 
    - To run ML pipeline: Run the following commands in the project's root directory:

      `python models/train_classifier.py data/etl_disaster.db models/cv_t.pkl

3. Open Workspace
   
4. Go to `app` directory: `cd app`

5. Run your web app: `python run.py`

6. Click the `PREVIEW` button to open the homepage
