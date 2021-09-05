# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:53:07 2021

@author: ruchuan2
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

### first I use the Treasury data

fin_data = pd.read_csv('//ad.uillinois.edu/engr-ews/ruchuan2/Desktop/Treasury Squeeze raw score data.csv',header='infer')
del fin_data['rowindex']
del fin_data['contract']

X = fin_data[['price_crossing','price_distortion']]
y = fin_data[['squeeze']]



# Splitting data into 70% training and 30% test data:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# Standardizing the features:

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# # K-nearest neighbors - a lazy learning algorithm

            
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')

knn.fit(X_train_std, y_train)

knn.score(X_test,y_test)


## Now, we test multiple different K inputs to try to identify the best choice of K for this data

## set the range of k
k_range = range(1, 30)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    
print(scores)

## [0.5148148148148148, 0.5555555555555556, 0.5518518518518518, 0.5888888888888889, 0.5703703703703704, 
## 0.5888888888888889, 0.5962962962962963, 0.6111111111111112, 0.5814814814814815, 0.6074074074074074, 
## 0.6185185185185185, 0.6111111111111112, 0.6222222222222222, 0.6444444444444445, 0.6148148148148148, 
## 0.6185185185185185, 0.6037037037037037, 0.6185185185185185, 0.6, 0.5814814814814815, 0.5925925925925926,
##  0.5777777777777777, 0.5777777777777777, 0.5703703703703704, 0.5777777777777777, 0.5777777777777777, 
## 0.5740740740740741, 0.5777777777777777, 0.5962962962962963]

## decision trees

# ## Maximizing information gain - getting the most bang for the buck
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# ## Building a decision tree

tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=5, 
                                    random_state=1)
tree_model.fit(X_train, y_train)

tree.plot_tree(tree_model)
plt.show()

## eavulate the decision tree
acc = accuracy_score(y_test, y_pred)
print(acc)

print("My name is Richie Ma")
print("My NetID is: ruchuan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
