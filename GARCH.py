#!/usr/bin/env python
# coding: utf-8

# # Generalized Autoregressive Conditional Heteroskedasticity.(GARCH)

# ##### The ARCH model is a statistical model for time series data that desribes the variance of a period t (current period) error term or innovation as a function of the actual sizes of the previous time periods' error terms. Often the variance is related to the square of previous innovations. The ARCH model is appropriate when the variance of the error term follows an AR process. However if the if an ARMA process is assummed to the error variance, then the model is a GARCH.

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
import statistics as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Let's import the data set
df1=pd.read_csv("C.csv",parse_dates=["T"], index_col=["T"])
df1.head()


# In[5]:


#Again lets take a look at the data.
pyplot.plot(df1["A"])
pyplot.show()


# In[6]:


#autocorellation plot
squared_data=df1["A"]**2
plot_acf(squared_data)
pyplot.show()


# #### Auto Regressive Conditional Heteroskedasticity. (ARCH)

# #Let's split the data series into training and test data.Here we have split the data into a training(2/3 of data) and a test set(1/3)of the data set

# In[7]:


train, test = df1.A[0:3748], df1.A[3748:-1]
# define model
model = arch_model(train, mean='Zero', vol='ARCH', p=15)
# fit model
model_fit = model.fit()
# forecast the test set
yhat = model_fit.forecast(horizon=5217)


# In[9]:


#Just taking a look at the sub dataset
df1.describe()
#d.mean(axis = 1, skipna = True)


# ## GARCH

# In[11]:


train, test = df1.A[0:3748], df1.A[3748:-1]
# define model
model1 = arch_model(train, mean="Zero", vol='GARCH', p=15)
# fit model
model1_fit = model1.fit()
# forecast the test set
yhat = model1_fit.forecast(horizon=len(df1["A"]))
#variance=sm.variance(df1["A"])
#pyplot.plot(variance)
#pyplot.show


# In[12]:


#fitting model results
eps = np.zeros_like(df1.A[0:100])
am = arch_model(eps)
res = am.fit(update_freq=1)
print(res.summary())


# In[ ]:




