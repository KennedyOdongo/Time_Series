#!/usr/bin/env python
# coding: utf-8

# #### Auto Regressive Integrated Moving Average.

# In[1]:


#import modules 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
register_matplotlib_converters()


# In[2]:


#read in the data set
df = pd.read_csv('F.csv')
df.head()


# In[3]:


#Lets clean the data set a bit. This code removes all the columns we do not need. In this case we only need: Australia, UK,Japanese
# and Euro. So we drop all the other columns.
drop_columns=["x","Y","C","H","M","I","P","D","S","MM","NN","SS","SR","SW","TW","TH","M.1","K"]
df.drop(drop_columns, inplace=True, axis=1)
df.head()


# In[4]:


#Force converting all the columns to numerical numbers. Afterwards converting the NaN's to zero's
df["A"]=pd.to_numeric(df["A"], errors='coerce')
df["E"]=pd.to_numeric(df["E"], errors='coerce')
df["U"]=pd.to_numeric(df["U"], errors='coerce')
df["J"]=pd.to_numeric(df["J"], errors='coerce')
df["A"] = df["A"].fillna(0)
df["E"] = df["E"].fillna(0)
df["U"] = df["U"].fillna(0)
df["J"] = df["J"].fillna(0)


# #### ADF check for stationarity. This has to be done before any prediction. If the data is not stationary we have to difference it to make it stationary

# In[5]:


#checking for stationarity using the dicky fuller test.
#For the Australian Series
result = adfuller(df['A'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
#For the Euro Series:

result1 = adfuller(df['E'])
print('ADF Statistic: {}'.format(result1[0]))
print('p-value: {}'.format(result1[1]))
print('Critical Values:')
for key, value in result1[4].items():
    print('\t{}: {}'.format(key, value))
    
#For the Japanese series:
result2 = adfuller(df['J'])
print('ADF Statistic: {}'.format(result2[0]))
print('p-value: {}'.format(result2[1]))
print('Critical Values:')
for key, value in result2[4].items():
    print('\t{}: {}'.format(key, value))

#For the UK pound series
result3 = adfuller(df['U'])
print('ADF Statistic: {}'.format(result3[0]))
print('p-value: {}'.format(result3[1]))
print('Critical Values:')
for key, value in result3[4].items():
    print('\t{}: {}'.format(key, value))


# ##### Clearly none of them are stationary from above. abs(dickerfuller stat)> p.values

# In[6]:


#Let's take a look at what our data set looks like.
df.describe()


# In[7]:


#This table only has the columns we need and the date column.
df.to_csv(r'C:\Users\Rodgers\Desktop\PhD courses\PhD courses\EconS 513 Jia Yan\C.csv', index = False)


# In[8]:


#Here we read in the data set with only the columns we need for this exercise
df1=pd.read_csv("C.csv",parse_dates=["T"], index_col=["T"])
df1.head()


# In[9]:


#As an example we plot the time series for Austalia to have a look at it
plt.plot(df1["A"])
plt.show()


# In[10]:


df_log = np.log(df["A"])
plt.plot(df_log)


# In[11]:


#plt.xlabel('Date')
#plt.ylabel('')
#plt.plot(df["J"])


# In[12]:


#let's difference the data to get a stationary
def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(df1['A'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


# In[13]:


#here we are plotting the graph for just one series.Australia.
rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
get_stationarity(df_log_minus_mean)


# In[14]:


#Arima model for "AUSTRALIA" timeseries.
model = ARIMA(df1["A"], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[15]:


#Arima model for "EURO" timeseries.
model = ARIMA(df1["E"], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[16]:


#Arima model for "JAPANESE" timeseries.
model = ARIMA(df1["J"], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[17]:


#Arima model for "UK" timeseries.
model = ARIMA(df1["U"], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[ ]:




