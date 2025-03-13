# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TVj7TngCbsx03R8yOxA1nFwQG0i-qqiu
"""

### import required libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

file_name="/pima-indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
DataFrame = read_csv(file_name, names=names)
DataFrame.head()

array=DataFrame.values
X=array[:,0:8]
Y=array[:,8]

## set the parameters
seed=7
kfold=KFold(n_splits=10,random_state=seed,shuffle=True)
cart=DecisionTreeClassifier()
num_trees=100
model=BaggingClassifier(estimator=cart,n_estimators=num_trees,random_state=seed)

reult=cross_val_score(model,X,Y,cv=kfold) # Changed k_fold to kfold
print(reult.mean()) # Changed result to reult to match the variable name in line 1

## ADA boost classifier

from sklearn.ensemble import AdaBoostClassifier
file_name="/pima-indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
DataFrame = read_csv(file_name, names=names)
DataFrame.head() # Changed dataframe to DataFrame to match the variable definition on the previous line
array=DataFrame.values # Changed dataframe to DataFrame to match the variable definition
X=array[:,0:8]
Y=array[:,8] # If you intended to add the values, use +=. If you intended assignment, use =.

num_trees=10
seed=7
k_fold=KFold(n_splits=10,random_state=seed,shuffle=True)
model=AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
result=cross_val_score(model,X,Y,cv=k_fold)
print(result.mean())

