#!/usr/bin/env python
# coding: utf-8

# Import Data set

# In[22]:


import pandas as pd
data = pd.read_csv(r"C:\Desktop\NewspaperData.csv")
data.head()


# In[23]:


data.min()


# In[24]:


data.max()


# In[25]:


data.info()


# In[26]:


data.duplicated()


# In[27]:


data.isnull().sum()


# In[28]:


data.shape


# In[29]:


data.sample(10)


# Corelation

# In[30]:


data.drop('Newspaper',axis=1).corr()


# In[31]:


import seaborn as sns
sns.distplot(data['daily'])


# Import Daat set

# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[34]:


import pandas as pd
data = pd.read_csv("NewspaperData.csv")
data.head()


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


sns.distplot(data['daily'])


# In[37]:


sns.distplot(data['sunday'])
plt.show()


# In[38]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data).fit()


# In[40]:


sns.regplot(x="daily", y="sunday",data=data,ci=None,color='black');#


# In[41]:


#Coefficients
model.params


# In[42]:


#t and p-Values
print(model.tvalues, '\n',model.pvalues)


# In[43]:


#R squared values
(model.rsquared,model.rsquared_adj)


# In[ ]:




