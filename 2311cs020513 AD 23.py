#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np


# In[18]:


import pandas as pd
cars = pd.read_csv(r"C:\Desktop\Cars.csv")
cars.head()


# In[19]:


cars.dtypes


# In[15]:


import warnings
warnings.filterwarnings('ignore')


# In[16]:


sns.distplot(cars["HP"])
plt.show()


# In[20]:


sns.distplot(cars["MPG"])
plt.show()


# In[27]:


sns.distplot(cars["HP"],color = 'black')
plt.show()


# In[23]:


cars.info


# In[28]:


sns.scatterplot(cars['HP'],color = 'black')
plt.show(1)


# In[31]:


sns.scatterplot(x=cars['HP'],y=cars['MPG'],color = 'black')
plt.show(1)


# In[32]:


from seaborn import scatterplot

scatterplot(x=cars['WT'],y=cars['MPG'])


# In[33]:


cars.corr()


# In[38]:


from seaborn import heatmap
heatmap(cars.corr(),color = 'black')


# In[ ]:




