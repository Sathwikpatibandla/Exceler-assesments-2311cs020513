#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Import .csv file and convert it to a DataFrame object
df = pd.read_csv(r"C:\Desktop\Sathwik\Wholesale customers data.csv")

df.head()


# In[3]:


df.head(1)


# In[4]:


df.tail()


# In[5]:


df.duplicated()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.Channel.unique()


# In[9]:


df.Channel


# In[10]:


df.describe()


# In[11]:


df.Region.unique()


# In[12]:


df.Region.count()


# In[13]:


import seaborn as sns


# In[14]:


sns.countplot(x=df['Channel'])
plt.show()


# In[15]:


import warnings
warnings.filterwarnings('ignore')


# In[16]:


sns.distplot(x=df['Channel'])


# In[17]:


sns.distplot(x=df['Fresh'])


# In[18]:


sns.distplot(x=df['Milk'])


# In[19]:


sns.distplot(x=df['Grocery'])


# In[20]:


sns.distplot(x=df['Detergents_Paper'])


# In[21]:


sns.distplot(x=df['Frozen'])


# In[22]:


sns.distplot(x=df['Delicassen'])


# In[23]:


df.info()


# In[24]:


df.min()


# In[25]:


df.max()


# In[26]:


df.min(1)


# In[27]:


df.max(1)


# In[28]:


df


# In[29]:


df.drop(['Channel','Region'],axis=1,inplace=True)


# In[30]:


df


# In[31]:


from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
X=stscaler.fit_transform(df)


# In[32]:


X


# In[33]:


import scipy.cluster.hierarchy as sch


# In[34]:


plt.figure(figsize=(20,6))
dendo = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer data')
plt.ylabel('Eucl Distance')
plt.show()


# In[35]:


len(set(dendo['color_list']))-1


# In[36]:


from sklearn.cluster import AgglomerativeClustering


# In[37]:


model = AgglomerativeClustering(n_clusters=3)
cluster=model.fit_predict(X)


# In[38]:


cluster


# In[40]:


cluster.shape


# In[41]:


df


# In[42]:


group_num=pd.DataFrame(cluster,columns=['Group'])
group_num


# In[43]:


pd.concat([df,group_num],axis=1)


# In[44]:


cust_group_data=pd.concat([df,group_num],axis=1)
cust_group_data


# # Kmeans

# In[45]:


X


# In[54]:


from sklearn.cluster import KMeans

wcss=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[55]:


wcss


# In[57]:


plt.plot(range(2,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




