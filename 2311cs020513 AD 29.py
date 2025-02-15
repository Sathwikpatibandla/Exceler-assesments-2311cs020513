#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[3]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import warnings
warnings.filterwarnings('ignore')


# In[4]:


titanic = pd.read_csv("C:\Desktop\Sathwik\Titanic.csv")


# In[5]:


titanic.head()


# In[6]:


titanic.tail()


# In[12]:


titanic['Class'].unique()


# In[13]:


titanic.describe()


# In[14]:


titanic.isnull().sum()


# In[15]:


titanic['Gender'].unique()


# In[17]:


pd


# In[19]:


titanic['Survived'].unique()


# In[20]:


titanic['Class'].unique()


# In[21]:


titanic['Age'].unique()


# In[22]:


titanic['Class'].value_counts()


# In[23]:


df=pd.get_dummies(titanic,dtype='int')
df.head()


# In[24]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

