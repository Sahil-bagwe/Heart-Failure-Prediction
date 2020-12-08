# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:36:15 2020

@author: Sahil Bagwe
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score,matthews_corrcoef,recall_score,precision_score
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 



df = pd.read_csv('C:/Users/Sahil Bagwe/Desktop/Python/dataset/heart.csv')
x = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking',]]

y = df[['DEATH_EVENT']]

#%%

ss = StandardScaler()
rs = RobustScaler()
mm = MinMaxScaler()

x =rs.fit_transform(x)


#%% Decision Tree

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3, random_state=103)
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train,Y_train)

# predictions = dtree.predict(X_test)

# cm = confusion_matrix(Y_test, predictions)
# mcc = matthews_corrcoef(Y_test,predictions)      
# print('\n')
# print('Decision Tree Accuracy:', accuracy_score(Y_test,predictions))
# print('Decision Tree f1-score: ', f1_score(Y_test, predictions))
# print('Decision Tree Precision:', precision_score(Y_test,predictions))
# print('Decision Tree Recall score: ', recall_score(Y_test, predictions))
# print('Decision Tree Mathews Coefficient: ',mcc)
# print('\n')
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g')
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Decision Tree Confusion Matrix')

#%%

# error_rate = []
# for i in range(1,1000,50):
#     model = RandomForestClassifier(n_estimators=i)
#     model.fit(X_train,Y_train.values.ravel())
#     predictions_i= model.predict(X_test)
#     error_rate.append(np.mean(predictions_i.reshape(90,1) != Y_test))

# plt.figure(figsize=(20,10))
# plt.xlabel("N- Estimators")
# plt.ylabel("Error Rate")
# plt.title("Error Rate for N-Estimators") 
# figure = plt.plot(range(1,1000,50),error_rate,color="blue",linestyle ="dashed",marker = "o",markerfacecolor = "red",markersize =20)

#%%
rfc = RandomForestClassifier(n_estimators = 400)
rfc.fit(X_train,Y_train.values.ravel())
rfc_predictions = rfc.predict(X_test)

cm = confusion_matrix(Y_test, rfc_predictions)
mcc = matthews_corrcoef(Y_test,rfc_predictions)      
print('\n')
print('Random Forest Classifier ', accuracy_score(Y_test,rfc_predictions))
print('Random Forest Classifier Precision:', precision_score(Y_test,rfc_predictions))
print('Random Forest Classifier Recall score: ', recall_score(Y_test, rfc_predictions))
print('Random Forest Classifier f1-score', f1_score(Y_test, rfc_predictions))
print('Random Forest Classifier MCC ',mcc)
# print(cm)

#%%
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Random Forest Confusion Matrix')