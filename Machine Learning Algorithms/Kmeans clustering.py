# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:37:41 2020

@author: Sahil Bagwe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,matthews_corrcoef,recall_score,precision_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA



df = pd.read_csv('C:/Users/Sahil Bagwe/Desktop/Python/dataset/heart.csv')
x = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking',]]


y = df[['DEATH_EVENT']]

#%% Data Scaling

# ss = StandardScaler()
# rs = RobustScaler()
# mm = MinMaxScaler()

# x = pd.DataFrame(data=rs.fit_transform(x))

#%% PCA


# pca = PCA().fit(x)
# # plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlabel('number of components')
# # plt.ylabel('cumulative explained variance');

# pca = PCA(n_components=3)
# x_pca = pca.fit_transform(x)

# x = pd.DataFrame(data = x_pca)

#%% Kmeans using PCA

kmeans = KMeans(n_clusters=2, random_state=666).fit(x)
train = x.copy()
train['cluster'] = kmeans.labels_
train['target'] = y
train

train['cluster'].value_counts()
mcc = matthews_corrcoef(train['target'], train['cluster'])

print('\n')
print('Kmeans Accuracy: ', accuracy_score(train['target'], train['cluster']))
print('Kmeans F1 Score: ', f1_score(train['target'], train['cluster']))
print('Kmeans Recall score: ', recall_score(train['target'], train['cluster']))
print('Kmeans Precision score', precision_score(train['target'], train['cluster']))
print('Kmeans MCC: ',mcc)


# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g')
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# # # error = []


#%% K Means Clustering

# kmeans = KMeans(n_clusters=2, random_state=666).fit(x)
# train = x.copy()
# train['cluster'] = kmeans.labels_
# train['target'] = y
# train

# train['cluster'].value_counts()

# print('Kmeans accuracy: ', accuracy_score(train['target'], train['cluster']))
# print('F1 Score: ', f1_score(train['target'], train['cluster']))
# cm = confusion_matrix(train['target'], train['cluster'])

# mcc = matthews_corrcoef(train['target'], train['cluster'])
# print("MCC:",mcc)

# print(cm)

# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g')

# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# # error = []


#%% Gaussian Mixture Model

# gmm = GMM(n_components=2).fit(x)
# gmm.fit_predict(x)
# gmm.score(x)