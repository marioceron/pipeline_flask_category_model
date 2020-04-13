import logging

import os
import pickle
import pandas as pd
import numpy as np
from flask import abort

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger('category')


class CategoryModel:
    category_model = None
    path = None

    category_name = ['business', 'entertainment', 'politics','sport','tech']

    def __init__(self, path=''):
        self.path = path
        self.category_model = None

        try:
            logger.info(path)
        except Exception as ex:
            logger.error(f"Error: {type(ex)} {ex}")
            abort(500)

    def load_resource(self, file_name):
        logger.info(file_name)
        return pickle.load(open(file_name, "rb"))

    def save_resource(self, resource, file_name):
        logger.info(file_name)
        pickle.dump(resource, open(file_name, "wb"))
    
    
    def train(self):
        status = -1
        try:
            
            #Read the  all the data collected on the notebook previously
            data_df = pd.read_csv("bbc_df.csv")

            # prepare the model data
            #X = data_df[['text_proc']]
            X = data_df[['text_count']]
            y = data_df.category
            #y = data_df.is_business

            # prepare the model and fit it

            #split in train and test set: 80/20, random_state=7
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)
            
            # Fit the model on training set
            #Logistic Regression : 
            category_model = LogisticRegression()

            #SGD Classifier:
            #category_model = SGDClassifier()

            #Multinomial Native Bayes:
            #category_model = MultinomialNB(fit_prior=True)

            #Random Forest Classifier
            #category_model=RandomForestClassifier()

            #Fit:
            category_model.fit(X_train, y_train)
            score=category_model.score(X_test, y_test)

            # test the model
            text="Ad sales boost Time Warner profit"
            text_count=1
            test_df = pd.DataFrame({'text':text},index=[0])

            predicted = category_model.predict(X=X_test)
            #predicted = category_model.predict(test_df.values)[0]
            #print("Predicted value=")
            print(predicted)
            logger.info("Predicted value: {}".format(predicted))
            
            # save the model
            self.save_resource(category_model, 'category.model')

            status = 'Train OK, ROC-AUC score = '+str(score)
            return status
        except Exception as ex:
            logger.error(f"Error: {type(ex)} {ex}")
            abort(500)


    def predict(self, parameters):
 
        try:
            logger.info(parameters)
            print(parameters)

            if not self.category_model:
                self.category_model = self.load_resource(os.path.join(self.path, 'category.model'))
            test_df = pd.DataFrame({'text':parameters},index=[0])
            text_count = test_df['text'].apply(lambda x: len(x.split(' ')))
            #text_count=340
            #print(text_count)
            count_df = pd.DataFrame({'text':text_count},index=[0])
            #print(count_df)

            predicted = self.category_model.predict(X=count_df)
            print(predicted)           
            logger.info("Predicted value: {}".format(predicted))
            #predicted_category = self.category_name[predicted]
            #print(predicted_category)
            #return predicted_category
            return str(predicted)
        except Exception as ex:
            logger.error(f"Error: {type(ex)} {ex}")
            abort(500)


