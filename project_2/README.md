# README


In this project I apply a classification algorithm to classify disaster messages.
Previously a cleaning step is performed, including extract the multiple targets, as well the creation of a database that will be the input of the classifier. 
The project has also a web app where an user can introduce messages and get the classification. 

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
  * process_data.py     # python script to process data and obtein a database. 
  * InsertDatabaseName.db   # database to save clean data to

- models folder
  * train_classifier.py  # python script to build a classification model and print the metrics. 
  * classifier.pkl  # saved model 

- README.md

Requirements
-----
This project runs under Python 3.9. Some packages must to be instaled. 

If you are using PyCharm just do the following: 
1. Create and environment in settings>project
2. Go to venv/Scripts and activate the environment by executing .\activate

In other cases, just run the scripts following the requirement expresses in the __main__
functions. That is to run the script followed by the necessary arguments as shown below:

```
python process_data.py messages_filepath, categories_filepath, database_filepath

python train_classifier.py database_filepath, model_filepath
```
