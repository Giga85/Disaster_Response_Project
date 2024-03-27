# Disaster_Response_Project
Disaster_Response_Project

### Instructions:

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
