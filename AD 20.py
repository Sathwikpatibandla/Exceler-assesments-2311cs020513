#!/usr/bin/env python
# coding: utf-8

# Import Data Set

# In[19]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


import pandas as pd
data = pd.read_csv(r"C:\Desktop\NewspaperData.csv")
data.head()


# In[ ]:





# In[2]:


data.info()


# In[3]:


data.head()


# In[13]:


data.sample()


# In[15]:


data.sample(15)


# Co Relation

# In[16]:


data.drop('Newspaper',axis=1).corr()


# In[17]:


import seaborn as sns
sns.displot(data['daily'])

