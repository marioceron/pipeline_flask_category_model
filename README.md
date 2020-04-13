# Introduction

This is an example of using Flask to create an API for an Python program. 

The service is exposing several endpoints as following:

* GET /status - return the status of the service
* GET train - train the model CategoryModel, from model module
* POST /category - 1 parameter: text of the article, then predict the category : (business, entertainment, politics, sport, tech) by calling the predict function of model module (model is LogisticRegression)

# Source files

The following files are included in the project:
* model/model.py - module containing the CategoryModel class: train the model, predict using the trianing model 
* app.py - API using Flask; to predict the category, a pretrained model (build in model.py using) is load

# Usage

To start the service, run `python app.py`