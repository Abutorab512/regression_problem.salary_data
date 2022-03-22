#!/usr/bin/env python
# coding: utf-8

# Import all the necessary library

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("D:\\Simplilearn all projects\\Data\\salary_data_file.csv")


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


x = data.drop(['Salary'],axis=1)
y = data['Salary']


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4587)


# In[7]:


print(x.shape,x_train.shape,x_test.shape)


# In[8]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


# In[9]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[10]:


model.coef_


# In[11]:


model.intercept_


# In[12]:


from sklearn.metrics import accuracy_score,r2_score
from math import sqrt
print(sqrt(r2_score(y_train,model.predict(x_train))))


# In[13]:


pd.DataFrame({'Actual':y_train,'Predicted':np.round(model.predict(x_train),0)}).head(10)

