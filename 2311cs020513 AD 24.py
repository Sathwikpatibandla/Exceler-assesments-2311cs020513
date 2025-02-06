#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np


# In[3]:


import pandas as pd
cars = pd.read_csv(r"C:\Desktop\Cars.csv")
cars.head()


# In[4]:


cars.dtypes


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


sns.distplot(cars["HP"])
plt.show()


# In[7]:


sns.distplot(cars["MPG"])
plt.show()


# In[8]:


sns.distplot(cars["HP"],color = 'black')
plt.show()


# In[9]:


cars.info


# In[10]:


sns.scatterplot(cars['HP'],color = 'black')
plt.show(1)


# In[11]:


sns.scatterplot(x=cars['HP'],y=cars['MPG'],color = 'black')
plt.show(1)


# In[12]:


from seaborn import scatterplot

scatterplot(x=cars['WT'],y=cars['MPG'])


# In[13]:


cars.corr()


# In[14]:


from seaborn import heatmap
heatmap(cars.corr(),color = 'black')


# In[16]:


sns.scatterplot(x=cars['VOL'],y=cars['MPG'])


# In[17]:


sns.scatterplot(x=cars['VOL'],y=cars['HP'])


# In[19]:


from seaborn import heatmap
heatmap(cars.corr())


# In[20]:


sns.scatterplot(x=cars['HP'],y=cars['SP'])


# In[21]:


sns.scatterplot(x=cars['HP'],y=cars['VOL'])


# In[22]:


sns.scatterplot(x=cars['HP'],y=cars['WT'])


# In[25]:


sns.scatterplot(x=cars['VOL'],y=cars['SP'])


# In[26]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])


# In[27]:


sns.scatterplot(x=cars['VOL'],y=cars['SP'])


# In[28]:


sns.scatterplot(x=cars['VOL'],y=cars['WT'])


# In[29]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])


# In[30]:


sns.scatterplot(x=cars['WT'],y=cars['HP'])


# In[31]:


sns.heatmap(cars.corr(),cmap='Blues',annot=True)
plt.show()


# In[33]:


sns.scatterplot(x=cars['SP'],y=cars['HP'])


# In[36]:


cars.info()


# In[37]:


sns.scatterplot(x=cars['SP'],y=cars['WT'])

