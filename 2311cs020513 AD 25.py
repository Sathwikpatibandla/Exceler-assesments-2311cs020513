#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


import pandas as pd
cars = pd.read_csv(r"C:\Desktop\Cars.csv")
cars.head()


# In[3]:


cars.dtypes


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


sns.distplot(cars["HP"])
plt.show()


# In[6]:


sns.distplot(cars["MPG"])
plt.show()


# In[7]:


sns.distplot(cars["HP"],color = 'black')
plt.show()


# In[8]:


cars.info


# In[9]:


sns.scatterplot(cars['HP'],color = 'black')
plt.show(1)


# In[10]:


sns.scatterplot(x=cars['HP'],y=cars['MPG'],color = 'black')
plt.show(1)


# In[11]:


from seaborn import scatterplot

scatterplot(x=cars['WT'],y=cars['MPG'])


# In[12]:


cars.corr()


# In[13]:


from seaborn import heatmap
heatmap(cars.corr(),color = 'black')


# In[14]:


sns.scatterplot(x=cars['VOL'],y=cars['MPG'])


# In[15]:


sns.scatterplot(x=cars['VOL'],y=cars['HP'])


# In[16]:


from seaborn import heatmap
heatmap(cars.corr())


# In[17]:


sns.scatterplot(x=cars['HP'],y=cars['SP'])


# In[18]:


sns.scatterplot(x=cars['HP'],y=cars['VOL'])


# In[19]:


sns.scatterplot(x=cars['HP'],y=cars['WT'])


# In[20]:


sns.scatterplot(x=cars['VOL'],y=cars['SP'])


# In[21]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])


# In[22]:


sns.scatterplot(x=cars['VOL'],y=cars['SP'])


# In[23]:


sns.scatterplot(x=cars['VOL'],y=cars['WT'])


# In[24]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])


# In[25]:


sns.scatterplot(x=cars['WT'],y=cars['HP'])


# In[26]:


sns.heatmap(cars.corr(),cmap='Blues',annot=True)
plt.show()


# In[27]:


sns.scatterplot(x=cars['SP'],y=cars['HP'])


# In[28]:


cars.info()


# In[29]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])


# In[30]:


cars.describe()


# In[34]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
cars[['HP','VOL','SP','WT']]


# Simple Linear Regression Models

# In[38]:


ml_v=smf.ols('MPG~VOL',data = cars).fit()
ml_v.rsquared


# In[39]:


ml_w=smf.ols('MPG~WT',data = cars).fit()
ml_w.rsquared


# In[40]:


ml_v.params


# In[41]:


ml_SP=smf.ols('MPG~SP',data = cars).fit()
ml_SP.rsquared


# In[42]:


cars.info()


# In[43]:


ml_HP=smf.ols('MPG~HP',data = cars).fit()
ml_HP.rsquared

