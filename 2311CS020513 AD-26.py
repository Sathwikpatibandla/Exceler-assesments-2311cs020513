#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pylot as plt


# In[7]:


# Import .csv file and convert it to a DataFrame object
df = pd.read_csv(r"C:\Desktop\Sathwik\Wholesale customers data.csv")

df.head()


# In[8]:


df.head(1)


# In[9]:


df.tail()


# In[10]:


df.duplicated()


# In[14]:


df.shape


# In[15]:


df.isnull().sum()


# In[16]:


df.Channel.unique()


# In[17]:


df.Channel


# In[18]:


df.describe()


# In[23]:


df.Region.unique()


# In[24]:


df.Region.count()


# In[30]:


import seaborn as sns


# In[32]:


sns.countplot(x=df['Channel'])
plt.show()


# In[38]:


import warnings
warnings.filterwarnings('ignore')


# In[39]:


sns.distplot(x=df['Channel'])


# In[40]:


sns.distplot(x=df['Fresh'])


# In[42]:


sns.distplot(x=df['Milk'])


# In[43]:


sns.distplot(x=df['Grocery'])


# In[44]:


sns.distplot(x=df['Detergents_Paper'])


# In[45]:


sns.distplot(x=df['Frozen'])


# In[46]:


sns.distplot(x=df['Delicassen'])


# In[47]:


df.info()


# In[50]:


df.min()


# In[51]:


df.max()


# In[52]:


df.min(1)


# In[54]:


df.max(1)


# In[ ]:




