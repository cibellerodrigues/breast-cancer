#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:50:47 2019

@author: cibelle
"""

import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criaRede():
    classificador = Sequential()
    classificador.add(Dense(units = 8, activation = 'relu', 
                            kernel_initializer = 'normal', input_dim = 30))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 8, activation = 'relu', 
                            kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 8, activation = 'tanh',
                           kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criaRede, epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador, 
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()
