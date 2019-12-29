# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:48:45 2019

@author: Remi_Adefioye
"""

#Load libraries
import numpy as np
import pandas as pd
#Load data
iris = pd.read_csv("iris.csv")

iris.head(15)
iris.tail(10)

#Label dataset as dependent and independent variables
X = iris.drop('species', axis =1)
y = iris['species']

#Load library for labeling
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

#View y
y

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units = 4, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))

#Last layer for multi-class classification of 3 species

model.add(Dense(units = 3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

model.fit(scaled_X, y, epochs = 150)

model.save("final_iris_model.h5")

import joblib
joblib.dump(scaler,'iris_scaler.pkl')