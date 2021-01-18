#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data=pd.read_csv(r'Downloads\shampoo.csv')
data.head()


# In[4]:


def data_prep(series,n_features):
    """ given a data series and the number of periods you want use to forecast, this function returns a value"""
    X,y=[],[]
    for i in range(len(series)):
        next_prediction=i+n_features
        if next_prediction> len(series)-1:
            break
        seq_x,seq_y=series[i:next_prediction],series[next_prediction]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X),np.array(y)


# In[5]:


X,y=data_prep(data['Sales'],3)


# In[6]:


X


# In[7]:


y


# #### Whenever you are using an LSTM you have to reshape your data into a 3 dimensional data set

# In[8]:


## We do a reshape from samples and time steps to samples, timestps and features 


# In[9]:


X=X.reshape((X.shape[0],X.shape[1],1))


# In[10]:


n_steps=3
n_features=1


# In[11]:


model=Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps,n_features)))
model.add(LSTM(50, activation='relu')) #layer with 50 neurons
model.add(Dense(1)) # we'll need only one output
model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=200,verbose=0) # Setting verbose to 1 shows you all the epochs


# In[12]:


x_input=np.array(data['Sales'][-3:])
temp_input=list(x_input)
lst_output=[]
i=0
while (i<20): # This depends on how far into the future you eant to predict, 10 periods ahead here
    if (len(temp_input)>3):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape((1, n_steps, n_features))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input=x_input.reshape((1, n_steps, n_features))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
print(lst_output)
    


# In[13]:


new=np.arange(1,21)
pred=np.arange(20,40)


# In[14]:


plt.plot(new,data["Sales"][-20:])
plt.plot(pred,lst_output)


# In[15]:


df=pdr.get_data_tiingo('AAPL',api_key='2616ae4c299d778e6ad7cc4a129f2a495322a5e8')


# In[16]:


df.to_csv('AAPL.csv')


# In[17]:


df1=df.reset_index()['close']


# In[18]:


df1.shape


# In[19]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[20]:


## LSTMS are very sensitive to the scales of the data


# In[21]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[22]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data, test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[23]:


def create_dataset(dataset, time_step=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+timestep,0])
    return np.array(dataX),np.array(dataY)


# In[24]:


timestep=100
xtrain,ytrain=create_dataset(train_data, timestep)
xtest,ytest=create_dataset(test_data,timestep)


# In[25]:


xtrain


# In[26]:


xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],1)


# In[27]:


model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True)) #layer with 50 neurons
model.add(LSTM(50))
model.add(Dense(1)) # we'll need only one output
model.compile(optimizer='adam',loss='mse')


# In[28]:


model.summary()


# In[29]:


model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=100,batch_size=64,verbose=0)


# In[30]:


train_predict=model.predict(xtrain)
test_predict=model.predict(xtest)


# In[31]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[32]:


math.sqrt(mean_squared_error(ytrain,train_predict))


# In[33]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[34]:


look_back=100
trainPredictPlot=np.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=np.empty_like(df1)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[35]:


len(test_data)


# In[36]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[37]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[38]:


lst_output=[]
n_steps=100
i=0
while(i<30): #for the next 30 days
    if (len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape(1,n_steps,1)
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i+=1
    else:
        x_input=x_input.reshape((1, n_steps, 1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i+=1
#print(lst_output)
        


# In[39]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[40]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

