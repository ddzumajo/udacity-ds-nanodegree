# README

This project 

Files
-----
The project is structure as shown below:
* app folder 
  * template 
  * master.html  # main page of web app
  * go.html  # classification result page of web app
  * run.py  # Flask file that runs app

- data folder
  * disaster_categories.csv  # data to process 
  * disaster_messages.csv  # data to process
  * process_data.py
  * InsertDatabaseName.db   # database to save clean data to

- models folder
  * train_classifier.py
  * classifier.pkl  # saved model 

- README.md

Requirements
-----
This project runs under Python 3.9. Some packaged must be included.

If you are using PyCharm just do the following: 
1. Create and environment in settings>project
2. Go to venv/Scripts and activate the environment by
executing.\activate
