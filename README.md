# Disaster Response Pipeline Project

### Overview
This repo uses disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.

In the repo, you'll find a data set containing real messages that were sent during disaster events. A machine learning pipeline can then be created to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Also include here is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files 
 - `disaster_categories.csv` and `disaster_messages.csv` contain messages received and their
 corresponding categories. These will act as training data for our model
  - `proess_data.py` : ETL pipeline that cleans data and stores in database
  - `train_classifier.py`  : ML pipeline that trains classifier and saves to pickle file
  -  `run.py` : script for runnning web app
   - `go.html` and `master.html` contain html for web app