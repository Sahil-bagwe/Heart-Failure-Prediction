# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:24:57 2020

@author: Sahil Bagwe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix,f1_score,matthews_corrcoef
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

df = pd.read_csv('C:/Users/Sahil Bagwe/Desktop/Python/dataset/heart.csv')
x = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking',]]


y = df[['DEATH_EVENT']]


#%% Scaler

ss = StandardScaler()
rs = RobustScaler()
mm = MinMaxScaler()

x =rs.fit_transform(x)

#%% Model

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=0)

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=0)
grid.fit(X_train,Y_train.values.ravel())
 
predictions = grid.predict(X_test)   
cm = confusion_matrix(Y_test, predictions)
mcc = matthews_corrcoef(Y_test,predictions)     

#%%
 
print('\n')
print('Serial Vector Machine Accuracy: ', accuracy_score(Y_test,predictions))
print('Serial Vector Machine f1-score:', f1_score(Y_test, predictions))
print('Serial Vector Machine Recall score: ', recall_score(Y_test,predictions))
print('Serial Vector Machine f1-score', f1_score(Y_test,predictions))
print('Serial Vector Machine MCC ',mcc)


#%%

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Serial Vector Machine Confusion Matrix')