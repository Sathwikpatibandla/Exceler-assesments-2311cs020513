#!/usr/bin/env python
# coding: utf-8

# In[27]:


from pandas import read_csv
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression

dataframe=sns.load_dataset('tips')
dataframe



# In[28]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
dataframe['smoker'] = lb.fit_transform(dataframe['smoker'])
dataframe['sex'] = lb.fit_transform(dataframe['sex'])
dataframe['time'] = lb.fit_transform(dataframe['time'])
dataframe['day'] = lb.fit_transform(dataframe['day'])


# In[29]:


x = dataframe.drop('tip',axis=1)
y = dataframe.tip


# In[30]:


x


# In[31]:


y


# In[34]:


test = SelectKBest(score_func=f_regression, k=3).fit(x,y)
test


# In[35]:


np.round(test.scores_,2)


# In[36]:


np.round(test.pvalues_,3)


# In[37]:


dataframe.columns[test.get_support(indices=True)]


# In[38]:


dataframe.columns[np.where(test.pvalues_>0.05)]


# In[39]:


df=sns.load_dataset('iris')


# In[40]:


df['species']=lb.fit_transform(df['species'])


# In[47]:


sel=SelectKBest(score_func=f_regression,k=2).fit(x,y)
sel


# In[48]:


sel.pvalues_


# In[49]:


sel.scores_


# In[51]:


dataframe.columns[np.where(sel.get_support(indices=True))]


# In[52]:


x


# In[53]:


y


# In[56]:


#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif,mutual_info_classif


# In[58]:


df=sns.load_dataset('iris')
df.head()

