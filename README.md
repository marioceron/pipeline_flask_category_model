# Pipeline Flask Category Model: 
Mario Alberto Ceron F. - DataScience Training April 2020

# Introduction

This is an example of using Flask to create an API for an Python program. 

https://github.com/marioceron/pipeline_flask_category_model

The service is exposing several endpoints as following:

* GET /status - return the status of the service
* GET train - train the model CategoryModel, from model module
* POST /category - 1 parameter: text of the article, then predict the category : (business, entertainment, politics, sport, tech) by calling the predict function of model module (model is LogisticRegression)

# Source files

The following files are included in the project:
* model/model.py - module containing the CategoryModel class: train the model, predict using the trianing model 
* app.py - API using Flask; to predict the category, a pretrained model (build in model.py using) is load
* category.model : Base Model used: Logistic Regression
* bbc_df.csv - Data extracted from BBC News website http://mlg.ucd.ie/files/datasets/bbc.zip
* pipeline_test_video/Video_pipeline_ds_test_run_marioceron.mp4. - Video demo of the pipeline app
* pipeline_test_video/Model.postman_collection.json. - Postman collection file with the data for testing (format json)
# Usage

To start the service, run `python app.py`

![Test Pipeline Running Example Politics Category](pipeline_running_example_politics_category.jpg)
