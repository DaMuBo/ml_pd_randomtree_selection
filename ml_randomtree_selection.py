"""
This publishs a class which performs feature selection where a random forest tree is used with standard parameters

"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split

class ml_FtrSelect_RandomTree():
    """
    Selects the remaining features from a standard Random Tree based of the threshold 
    Variables:
    --------------------------
    
    
    Methods:
    --------------------------
    
    
    --------------------------
    """
    
    def __init__(self, df=None, x=None,y=None, typ='Regressor', threshold = 0.0, trainsplit=0.8):
        """
        Initializing the Object and saving the initial variables
        Inputs:
        ------------
        df: DataFrame,
            the dataframe which holds the data
        x: list
            the list of attributes which will be given to the modell for the learning 
        y: str
            the target attribute whilc will be predicted
        typ: String
            the type of prediction performed. You can Perform a regression or a classification on the data .
            Possible Values = 'Regressor' or 'Classifier'
            Default Value = 'Regressor'
        threshold: float
            the threshold. a data with a feature importance above this threshold will be kept everything else will be deleted
            Default Value = 0.0
        trainsplit: float
            The amount of data which will be kept back for the test evaluation. Standard is 0.8 if you set this value on 1 there will be no data tested and it won't be possible to call the metrics on the modell
        """
        self.x = x
        self.y = y
        self.typ = typ
        self.threshold = threshold
        self.trainsplit = trainsplit
        
        self.features = None
    
    def fit(self, df, x=None,y=None,typ='Regressor', threshold = 0.0, trainsplit=0.8):
        """
        fitting new data to the model and train it
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.typ = typ
        self.threshold = threshold
        self.trainsplit = trainsplit
        
        
        # train test split
        if self.trainsplit < 1:
            df_train, df_test = train_test_split(df, train_size=self.trainsplit)
            X_train, y_train = df_train[self.x], df_train[self.y]
            X_test, y_test = df_test[self.x], df_test[self.y]
        else:
            X_train, y_train = df[self.x], df[self.y]
            
        if self.typ == 'Regressor':
            clf = RandomForestRegressor()
        elif self.typ == 'Classifier':
            clf = RandomForestClassifier()
        else:
            logging.warning("== No valid typ defined. Please check documentation for allowed types for the fit ==")
        clf.fit(X_train, y_train)
        sorted_idx = clf.feature_importances_.argsort()
        features = dict(zip(clf.feature_names_in_[sorted_idx], clf.feature_importances_[sorted_idx]))
        
        self.features = [[key, features[key]] for key in features if features[key] > threshold and key != y]
        if len(self.features) == 0:
            logging.warning(f"== There are no features left with a feature importance >  {threshold} ==")
        return self
    

    def transform(self,df):
        """
        transform new data and reduce the dataframe to the given columns
        """
        for n in self.features:
            liste.append(n[0])
        data = df[liste + [self.y]]
        
        return data

    
    
    def save_model(self,filename):
        """
        speichert die gelernte Struktur in einem File ab für die weitere Verwendung in einem 
        Produktivem Umfeld
        """
        output = {
            'x':self.x
            , 'y':self.y
            , 'threshold':self.threshold
            , 'typ':self.typ
            , 'trainsplit': self.trainsplit
            , 'self.features': self.features
        }
        pickle.dump(output, open(filename, 'wb'))
        
        
        
        
    def load_model(self, link):
        """
        lädt eine Struktur in das Objekt und gibt dieses danach zurück
        """
        inputi = pickle.load(open(link, 'rb'))
        self.x = inputi['x']
        self.y = inputi['y']
        self.threshold = inputi['threshold']
        self.typ = inputi['typ']
        self.trainsplit = inputi['trainsplit']
        self.features = inputi['features']

        return self
    
