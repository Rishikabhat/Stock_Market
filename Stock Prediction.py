#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[16]:


dataset=pd.read_csv('NseTataDataset.csv')


# In[17]:


dataset.describe()


# In[18]:


x=dataset[['High','Low','Open','Total Trade Quantity']].values
y=dataset['Close'].values


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[20]:


regressor=LinearRegression()


# In[21]:


regressor.fit(x_train,y_train)


# In[22]:


print(regressor.coef_)


# In[23]:


print(regressor.intercept_)


# In[24]:


predicted= regressor.predict(x_test)


# In[25]:


print(predicted)


# In[26]:


dframe=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predicted.flatten()})


# In[27]:


dframe.head(30)


# In[36]:


plt.scatter(predicted,y_test)
plt.plot(predicted,y_test,color='pink')
plt.show()


# In[17]:


print('Mean absolute Error:',metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,predicted)))


# In[18]:


import math


# In[19]:


graph=dframe.head(20)


# In[20]:


graph.plot(kind='bar')


# In[21]:


plt.plot(dframe)
plt.title('line')
plt.xlabel('x')
plt.ylabel('y')     
plt.show()


# In[ ]:





# In[ ]:




