#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[19]:


df=sns.load_dataset('iris')
df


# In[20]:


df.info()


# In[21]:


pd.get_dummies(df['species'])


# In[22]:


df['species'].unique()


# In[23]:


df['sepal_length'].unique()


# In[24]:


df['petal_length'].unique()


# In[25]:


df


# In[29]:


pd.concat([df1,df2])


# In[27]:


oh_species=pd.get_dummies(df['species'])
df=pd.concat([df.drop('species',axis=1),oh_species],axis=1)


# In[28]:


df


# In[30]:


from scipy.stats import zscore

data=sns.load_dataset('iris')
df=data.copy()
z_scores = np.abs(zscore(df.drop('species',axis=1)))


# In[31]:


z_scores


# In[32]:


zscore(df.drop('species',axis=1))


# In[39]:


threshold = 3

non_outliers = (z_scores < threshold).all(axis=1)
# Filter the DataFrame to keep only rows that are not outlier

df_no_outliers = df[non_outliers]


# In[40]:


df_no_outliers


# In[41]:


threshold = 3

outliers = (z_scores > threshold)

outliers_rows = df[outliers.any(axis=1)]
outliers_rows


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


from sklearn.ensemble import IsolationForest
import seaborn as sns
import pandas as pd
import numpy as np


# In[43]:


data =sns.load_dataset("iris")
data


# In[44]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['species']=lb.fit_transform(data['species'])


# In[45]:


clf = IsolationForest(random_state=10,contamination=.01)
clf.fit(data)


# In[47]:


clf.predict(data)


# In[53]:


y_pred_putliers=clf.predict(data)


# In[54]:


np.where(y_pred_outliers==-1)



# In[57]:


y_pred_outliers = clf.predict(data)
y_pred_outliers==-1


# In[58]:


data.drop(index=[117,131],axis=0)


# In[ ]:




