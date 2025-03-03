#!/usr/bin/env python
# coding: utf-8

# In[3]:


##install the modules
get_ipython().system('pip install streamlit')


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix


# In[24]:


data=pd.read_csv(r"C:\Desktop\Sathwik\Heart_Disease_Dataset.csv")


# In[25]:


data


# In[26]:


data.head()


# In[27]:


data.tail()


# In[28]:


data.shape


# In[29]:


data.info()


# In[30]:


data.describe()


# In[31]:


data.isnull().sum()


# In[32]:


data


# In[33]:


data.duplicated()


# In[34]:


len(data[data.duplicated()])


# In[35]:


data.isnull()


# In[36]:


##
for column in data.columns:
    print(data[column].value_counts())


# In[37]:


lis=[2,34,4,5,6]
for i in lis:
    print(i)


# In[ ]:





# In[ ]:




