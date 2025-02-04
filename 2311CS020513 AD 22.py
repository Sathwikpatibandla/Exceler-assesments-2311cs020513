#!/usr/bin/env python
# coding: utf-8

# Import Data set

# In[1]:


import pandas as pd
data = pd.read_csv(r"C:\Desktop\NewspaperData.csv")
data.head()


# In[2]:


data.min()


# In[3]:


data.max()


# In[4]:


data.info()


# In[5]:


data.duplicated()


# In[6]:


data.isnull().sum()


# In[7]:


data.shape


# In[8]:


data.sample(10)


# Corelation

# In[9]:


data.drop('Newspaper',axis=1).corr()


# In[13]:


import seaborn as sns
sns.distplot(data['daily'])


# Import Daat set

# In[14]:


import warnings
warnings.filterwarnings('ignore')


# In[15]:


import pandas as pd
data = pd.read_csv("NewspaperData.csv")
data.head()


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


sns.distplot(data['daily'])


# In[18]:


sns.distplot(data['sunday'])
plt.show()


# In[19]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data).fit()


# In[20]:


sns.regplot(x="daily", y="sunday",data=data,ci=None,color='black');#


# In[21]:


#Coefficients
model.params


# In[22]:


#t and p-Values
print(model.tvalues, '\n',model.pvalues)


# In[23]:


#R squared values
(model.rsquared,model.rsquared_adj)


# In[24]:


model.params


# In[ ]:


Y=mX +C

m= coef of x
c= constant intercept


# In[25]:


daily=200


# In[26]:


sunday=1.339715*daily + 13.835630


# In[27]:


sunday+37.35


# In[30]:


sunday


# In[31]:


#R squared values
model.rsquared


# In[32]:


sunday


# In[33]:


sunday+108

# Predict for new data point
# In[35]:


#Predict for 1000 and 3000 daily circulation
newdata=pd.Series([1000,3000,500,1500,4000])
newdata


# In[36]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[37]:


newdata.shape


# In[38]:


data_pred.shape


# In[39]:


data.info()


# In[40]:


model.predict(data_pred)


# In[41]:


data_pred


# In[42]:


from sklearn.linear_model import LinearRegression


# In[46]:


data.head()


# In[49]:


lr=LinearRegression()
lr.fit(data[['daily']],data['sunday'])


# In[57]:


data['daily']


# In[58]:


lr.predict(data_pred[['daily']])


# In[61]:


lr.predict(data_pred)


# In[63]:


lr.coef_


# In[64]:


lr.intercept_

