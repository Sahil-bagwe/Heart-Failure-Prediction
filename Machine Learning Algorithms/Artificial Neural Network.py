# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:11:51 2020

@author: Sahil Bagwe
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#%%  Data Set 

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

#%%

ratio = 95/ (95+299)
weights = [ratio, 1.0 - ratio]

#%%

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3, random_state=103)

model = Sequential()
model.add(Dense(11,activation="sigmoid"))
model.add(Dense(6,activation="sigmoid"))
model.add(Dense(1))
model.compile(optimizer='rmsprop',loss = "binary_crossentropy",metrics=["BinaryAccuracy"],loss_weights=weights)
history = model.fit(x=X_train,y=Y_train,epochs=450)

predictions = model.predict_classes(X_test)    

#%%

cm = confusion_matrix(Y_test, predictions)
mcc = matthews_corrcoef(Y_test,predictions)      
print('\n')
print('Neural Network Accuracy: ', accuracy_score(Y_test,predictions))
print('Neural Network f1-score:', f1_score(Y_test, predictions))
print('Neural Network Recall score: ', recall_score(Y_test,predictions))
print('Neural Network f1-score', f1_score(Y_test,predictions))
print('Neural Network MCC ',mcc)
#%%

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Neural Network Confusion Matrix')