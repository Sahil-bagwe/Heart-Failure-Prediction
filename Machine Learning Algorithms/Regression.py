# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:12:27 2020

@author: Sahil Bagwe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv('C:/Users/Sahil Bagwe/Desktop/Python/dataset/heart.csv')
x = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
# x = sc.fit_transform(x)
y = df[['DEATH_EVENT']]

#%% PCA

ss = StandardScaler()
rs = RobustScaler()
mm = MinMaxScaler()

x=rs.fit_transform(x)

#%% Model

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)


lr = LogisticRegression(random_state=111)
lr.fit(X_train,Y_train.values.ravel())
preds = lr.predict(X_test)

cm = confusion_matrix(Y_test, preds)
mcc = matthews_corrcoef(Y_test,preds) 

print('\n')
print('Logistic Regression Accuracy: ', accuracy_score(Y_test,preds))
print('Logistic Regression f1-score:', f1_score(Y_test, preds))
print('Logistic Regression Recall score: ', recall_score(Y_test,preds))
print('Logistic Regression f1-score', f1_score(Y_test,preds))
print('Logistic Regression MCC ',mcc)


ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Logistic Regression Confusion Matrix')




