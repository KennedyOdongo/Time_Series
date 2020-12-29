#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import base
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
import warnings
warnings.filterwarnings('ignore')
from itertools import chain
from tqdm import tqdm
from collections.abc import Mapping, Sequence, Iterable
from itertools import product
from functools import partial, reduce
import operator


# #### Data preparation.

# In[3]:


data=pd.read_csv('Sales.csv')
data.head()


# #### Melt the dataset to convert it from a wide to a long format.

# In[4]:


df=data.melt(id_vars='Product_Code', var_name='Week',value_name='Sales')
df['Product_Code']=df['Product_Code'].str.extract('(\d+)', expand=False).astype(int) # extract the integer portions of each column
df['Week'] = df['Week'].str.extract('(\d+)', expand=False).astype(int)
df.head() # visualize conversion from a wide to a long format.


# ####  Scikit-Learn provides us with two great base classes, TransformerMixin and BaseEstimator. Inheriting from TransformerMixin ensures that all we need to do is write our fit and transform methods and we get fit_transform for free. Inheriting from BaseEstimator ensures we get get_params and set_params for free.

# #### To convert a time series prediction problem into ML problem we create a supervised learning model in which previous lags are used as predictors of future output. 

# ### Create Supervised and Differenced Classes

# In[5]:


class Supervised(BaseEstimator,TransformerMixin): #inheriting from the sklearn's base class
    def __init__(self,col,groupCol,numLags,dropna=False):
        self.col=col
        self.groupCol=groupCol
        self.numLags=numLags
        self.dropna=dropna
    def fit(self,X,y=None):
        self.X=X
        return self
    def transform(self,X):
        copy=self.X.copy()
        for i in range(1,self.numLags+1):
            copy[str(i)+"_Week_Ago"+"_"+self.col]=copy.groupby([self.groupCol])[self.col].shift(i) # shift function to create lags
        if self.dropna:
            copy=copy.dropna()
            copy=copy.reset_index(drop=True)
        return copy
        


# In[6]:


class Difference(BaseEstimator,TransformerMixin):
    def __init__(self,col,groupCol,numLags,dropna=False):
        self.col=col
        self.groupCol=groupCol
        self.numLags=numLags
        self.dropna=dropna
    def fit(self,X,y=None):
        self.X=X
        return self
    def transform(self,X):
        copy=self.X.copy()
        for i in range(1,self.numLags+1):
            copy[str(i)+'_Week_Ago_Diff_'+'_'+self.col]=copy.groupby([self.groupCol])[self.col].diff(i) #diff function to get a differenced time series.
        if self.dropna:
            copy=copy.dropna()
            copy=copy.reset_index(drop=True)
        return copy


# #### K-Fold

# In[7]:


class Kfold():
    def __init__(self,**options):
        self.target=options.pop('target',None)
        self.date_col=options.pop('date_col',None)
        self.date_init=options.pop('date_init',None)
        self.date_final=options.pop('date_final',None)
        
        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))
        if ((self.target==None)|(self.date_col==None)|(self.date_init==None)| (self.date_final==None)):
            raise TypeError("Insufficient Arguments")
    def train_test_split(self,X):
        no_arrays=len(X)
        if no_arrays==0:
            raise ValueError ("Supply at least one array as an input")
            
        for i in range(self.date_init,self.date_final):
            
            train = X[X[self.date_col] < i]
            val   = X[X[self.date_col] == i]
            
            X_train, X_test=train.drop([self.target],axis=1),val.drop([self.target],axis=1)
            y_train, y_test=train[self.target].values,val[self.target].values
            
            yield X_train,X_test,y_train,y_test
            
    def split(self,X):
        train_cv=self.train_test_split(X)
        return chain(train_cv)
            


# #### The yield statement suspends execution of a function and sends a value back to caller, while saving state and later resuming meaning the whole generator itself can still be resumed after the return value is obtained. A return statement ends the execution of the function and sends a value back to the caller.

# ### Evaluation Metrics

# In[8]:


def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))


# ## Toy Model.

# In[9]:


class Toymodel(BaseEstimator,RegressorMixin):
    def __init__ (self,predCol):
         self.predCol = predCol
    def fit(self,X,y):
        return self
    
    def predict(self,X):
        prediction=X[self.predCol].values ## Toy model assumes that this week's sales are the same as last weeks values
        return prediction   
    def score(self,X,y,scoring):
        prediction=self.predict(X)
        error=scoring(y,prediction)
    
        return error
        


# #### Real values and Log Regressions:Time Series

# In[10]:


class TSReg(BaseEstimator,RegressorMixin):
    def __init__ (self, model,cv,scoring,verbosity=True):
        self.model=model
        self.cv=cv
        self.verbosity=verbosity
        self.scoring=scoring
        
    def fit (self,X,y=None):
        return self
    
    def predict(self,X=None):
        pred={}
        for indx, fold in enumerate(self.cv.split(X)):
            X_train,X_test,y_train,y_test=fold
            self.model.fit(X_train,y_train)
            pred[str(indx)+'_fold']=self.model.predict(X_test)
            prediction=pd.Dataframe(pred)
            
            return prediction
        
    def score(self,X,y=None):
        errors=[]
        for indx,fold in enumerate(self.cv.split(X)):
            X_train,X_test,y_train,y_test=fold
            self.model.fit(X_train,y_train)
            prediction = self.model.predict(X_test)
            error = self.scoring(y_test, prediction)
            errors.append(error)
            
            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
            print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors
            
            


# In[11]:


class TSLog(BaseEstimator,RegressorMixin):
    def __init__ (self, model,cv,scoring,verbosity=True):
        self.model=model
        self.cv=cv
        self.verbosity=verbosity
        self.scoring=scoring
        
    def fit (self,X,y=None):
        return self
    
    def predict(self,X=None):
        pred={}
        for indx, fold in enumerate(self.cv.split(X)):
            X_train,X_test,y_train,y_test=fold
            self.model.fit(X_train,y_train)
            pred[str(indx)+'_fold']=self.model.predict(X_test)
            prediction=pd.Dataframe(pred)
            
            return prediction
        
    def score(self,X,y=None):
        errors=[]
        for indx,fold in enumerate(self.cv.split(X)):
            X_train,X_test,y_train,y_test=fold
            self.model.fit(X_train,np.log1p(y_train))
            prediction = np.expm1(self.model.predict(X_test))
            error = self.scoring(y_test, prediction)
            errors.append(error)
            
            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
            print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors
            
            


# In[12]:


def GetPipeline(i):
    steps=[(str(i)+'_step',Supervised('Sales','Product_Code',i))]
    for j in range(1,i+1):
        if i==j:
            pp=(str(j)+'_step_diff', Difference(str(i)+'_Week_Ago_Sales','Product_Code',1,dropna=True))
            steps.append(pp)
            
        else:
            pp=(str(j)+'_step_diff', Difference(str(i)+'_Week_Ago_Sales','Product_Code',1))
            steps.append(pp)
            
    return steps


# In[13]:


def Tune(X, model,num_steps, init=1):
    scores=[]
    
    for i in tqdm(range(init,num_steps+1)):
        steps=[]
        steps.extend(GetPipeline(i))
        steps.append(('predict_1',model))
        super_=Pipeline(steps).fit(X)
        score_np.mean(super_.score(X))
        scores.append((i, score_))
    return scores


# ### Hyperparameter Tuning

# In[14]:


class ToyGrid(BaseEstimator,RegressorMixin):
    def __init__(self,param_grid):
        if not isinstance(param_grid,(Mapping, Iterable)):
            raise TypeError('Grid of parameters is not a dict or a list({!r})'.format(param_grid))
        if isinstance (param_grid,Mapping):
            param_grid=[param_grid]
        
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Grid of parameters is not a dict ({!r})'.format(param_grid))
            for key in grid:
                if not isinstance(grid[key],Iterable):
                    raise TypeError('Parameter grid value is not iterable(key=({!r})),value=({!r})'.format(key,grid[key]))
        self.param_grid=param_grid
        
        
    def __iter__ (self):
        for p in self.paramgrid:
            items =sorted(p.items())
            if not items:
                yield {}
            else:
                keys,values=zip(*items)
                for v in product(*values):
                    params=dict(zip(keys,v))
                    yield params
                    


# In[15]:


class TSGrid(ToyGrid,BaseEstimator,RegressorMixin):
    def __init__(self,**options):
        self.model=options.pop('model',None)
        self.cv=options.pop('cv',None)
        self.verbosity=options.pop('verbosity',False)
        self.scoring=options.pop('scoring',None)
        self.param_grid=ToyGrid(param_grid)
        
        if options:
            raise TypeError("Invalid parameters passed: %s"% str(options))
            
        if ((self.model==None)|(self.cv==None)):
            raise TypeError("Incomplete inputs")
            
    def fit(self,X,y=None):
        self.X=X
        return self
    
    def get_score(self,X,y=None):
        errors=[]
        for indx,fold in enumerate(self.cv.split(X)):
            X_train,X_test,y_train,y_test=fold
            self.model.set_params(**param).fit(X_train,np.log1p(y_train))
            prediction = np.expm1(self.model.predict(X_test))
            error = self.scoring(y_test, prediction)
            errors.append(error)
            
            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
            print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors
    def score(self):
        errors=[]
        get_param=[]
        
        for param in self.param_grid:
            
            if self.verbosity:
                print(param)
            errors.append(np.mean(self.get_score(param)))
            get_param.append(param)
        self.sorted_errors,self.sorted_params = (list(t) for t in zip(*sorted(zip(errors,get_param))))
        
        return self.sorted_errors,self.sorted_params
    
    def best_estimator(self,verbosity=False):
        if verbosity:
            print('error:{:.4f}\n'.format(self.sorted_errors[0]))
            print('Best params:')
            print(self.sorted_params[0])
        return self.sorted_params[0]
            


# ### Vizualizing the data

# In[16]:


df['Sales'].hist(bins=20, figsize=(10,5))
plt.xlabel('Number of Sales',fontsize=14)
plt.ylabel('Count Number of Sales',fontsize=14)
plt.title('Sales',fontsize=20)
plt.show()


# #### Data skewed to the left we might need a transformation for the model to perform better.

# #### Data prep

# In[18]:



steps = [('1_step',Supervised('Sales','Product_Code',1)),
         ('1_step_diff',Difference('1_Week_Ago_Sales','Product_Code',1,dropna=True))]
super_1 = Pipeline(steps).fit_transform(df)


# In[19]:


super_1.head()


# ## Toy Model.

# In[20]:


kf = Kfold(target='Sales',date_col = 'Week',date_init=42,date_final=52)


# In[22]:


base_model = Toymodel('1_Week_Ago_Sales')
errors = []
for indx,fold in enumerate(kf.split(super_1)):
    X_train, X_test, y_train, y_test = fold
    error = base_model.score(X_test,y_test,rmsle)
    errors.append(error)
    print("Fold: {}, Error: {:.3f}".format(indx,error))
    
print('Total Error {:.3f}'.format(np.mean(errors)))


# ## Lag 1

# In[23]:


model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)


# In[24]:


steps_1 = [('1_step',Supervised('Sales','Product_Code',1)),
         ('1_step_diff',Difference('1_Week_Ago_Sales','Product_Code',1,dropna=True)),
         ('predic_1',TSReg(model=model,cv=kf,scoring=rmsle))]
super_1_p = Pipeline(steps_1).fit(df)


# In[27]:


Model_1_Error = super_1_p.score(df)


# In[ ]:




