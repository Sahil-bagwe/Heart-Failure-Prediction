# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:44:01 2020

@author: Sahil Bagwe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,matthews_corrcoef

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

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=0)

error_rate = []
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    predictions_i= model.predict(X_test)
    error_rate.append(np.mean(predictions_i.reshape(90,1) != Y_test))

# plt.figure(figsize=(20,10))
# plt.xlabel("K value")
# plt.ylabel("Error Rate")
# plt.title("Error Rate for K-Neightbors") 
# figure = plt.plot(range(1,40),error_rate,color="blue",linestyle ="dashed",marker = "o",markerfacecolor = "red",markersize =20)

#%%

model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train,Y_train)
predictions= model.predict(X_test)

#%%
cm = confusion_matrix(Y_test, predictions)
mcc = matthews_corrcoef(Y_test,predictions)     

print('\n')
print('KNN Accuracy:', accuracy_score(Y_test,predictions))
print('KNN f1-score: ', f1_score(Y_test, predictions))
print('KNN Precision:', precision_score(Y_test,predictions))
print('KNN Recall score: ', recall_score(Y_test, predictions))
print('KNN Mathews Coefficient: ',mcc)
print('\n')
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('KNN Confusion Matrix')