#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 00:01:06 2022

@author: moeezkhan
"""

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from collections import Counter


# This function mutates, and also returns, the targetDF DataFrame.
# Mutations are based on values in the sourceDF DataFrame.
# You'll need to write more code in this function, to complete it.
def preprocess(targetDF, sourceDF):
    #For the Sex attribute, replace all male values with 0, and female values with 1.
    #(For this historical dataset of Titanic passengers, only "male" and "female" are listed for sex.)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 0 if v == "male" else v)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 1 if v == "female" else v)
    
    # Fill not-available age values with the median value.
    targetDF.loc[:, 'Age'] = targetDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    
	# -------------------------------------------------------------
	# Problem 4 code goes here, for fixing the error
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].fillna(sourceDF.loc[:, "Embarked"].mode()[0])
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 0 if v =='C' else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 1 if v =='Q' else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 2 if v =='S' else v)
    

    # -------------------------------------------------------------
	# Problem 5 code goes here, for fixing the error
    targetDF.loc[:, "Fare"] = targetDF.loc[:, "Fare"].fillna(sourceDF.loc[: , "Fare"].median())

	

# You'll need to write more code in this function, to complete it.
def buildAndTestModel():
    titanicTrain = pd.read_csv("data/train.csv")
    preprocess(titanicTrain, titanicTrain)
	
	# -------------------------------------------------------------
	# Problem 4 code goes here, to make the LogisticRegression object.
    model = LogisticRegression(solver = 'liblinear')
    inputCols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    outputCol = "Survived"
        
    inputDF = titanicTrain.loc[: , inputCols]
    outputSeries = titanicTrain.loc[: , outputCol] 
    
    cvScores = model_selection.cross_val_score(model, inputDF, outputSeries, cv = 3, scoring='accuracy')
    averageAccuracy = np.mean(cvScores)
    print(averageAccuracy)
    
	
	# -------------------------------------------------------------
	# Problem 5 code goes here, to try the Kaggle testing set
    titanicTest = pd.read_csv("data/test.csv")
    preprocess(titanicTest, titanicTrain)
    
    testInputDF = titanicTest.loc[:, inputCols]    
    
    model2 = LogisticRegression(solver = 'liblinear')
    model2.fit(inputDF, outputSeries)
    predictions = model2.predict(testInputDF)
    print(predictions, Counter(predictions), sep="\n")
    
    
    submitDF = pd.DataFrame({"PassengerId": titanicTest.loc[:, "PassengerId"], 
                             "Survived": predictions})
    submitDF.to_csv("data/submission.csv", index=False)


	
def test06():
    buildAndTestModel()    
    
    
test06()
