#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


# In[2]:


class FeatureSelector( BaseEstimator, TransformerMixin ):
    def __init__(self,feature_names):
        self._feature_names=feature_names
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self._feature_names] # returns a pandas dataframe with only the slected columns


# ##### one hot encoder which returns a dense representation of our pre-processed data

# ## categorical transformer

# In[4]:


class CategoricalTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, use_dates=['year','month','day']):
        self._use_dates=use_dates
    
    def fit(self,X,y=None):
        return self
    
    def get_year( self, obj ):
        return str(obj)[:4] #Extract year from the time stamp.
    
    def get_month( self, obj ):
        return str(obj)[4:6] # Extract month from the time stamp.
    
    def get_day(self, obj):
        return str(obj)[6:8] # Extract day from the time stamp.
    
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'
    def transform(self, X , y = None ): 
        for spec in self._use_dates:
            exec( "X.loc[:,'{}'] = X['date'].apply(self.get_{})".format( spec, spec ) )
            #Drop unusable column 
            X = X.drop('date', axis = 1 )
       
            #Convert these columns to binary for one-hot-encoding later
            X.loc[:,'waterfront'] = X['waterfront'].apply( self.create_binary )
       
            X.loc[:,'view'] = X['view'].apply( self.create_binary )
       
            X.loc[:,'yr_renovated'] = X['yr_renovated'].apply( self.create_binary )
            #returns numpy array
            return X.values 


# ## numerical transformer

# In[5]:


class NumericalTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,bath_per_bed=True, years_old=True):
        self._bath_per_bed=bath_per_bed
        self._years_old=years_old
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        
        if self._bath_per_bed:
            X.loc[:,'bath_per_bed']=X['bathrooms']/X['bedrooms']
            X.drop('bathrooms',axis=1)
        
        if self._years_old:
            
            X.loc[:,'years_old']=2021-X['yr_built']
            X.drop(yr_built,axis=1)
            
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        return X.values


# #### Now that we’ve written our numerical and categorical transformers and defined what our pipelines are going to be, we need a way to combine them, horizontally. We can do that using the FeatureUnion class in scikit-learn. Concatenates the results of multiple transformer objects

# In[7]:


#Categrical features to pass down the categorical pipeline 
categorical_features = ['date', 'waterfront', 'view', 'yr_renovated']

#Numerical features to pass down the numerical pipeline 
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'condition', 'grade', 'sqft_basement', 'yr_built']

#Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ),
                                  
                                  ( 'cat_transformer', CategoricalTransformer() ), 
                                  
                                  ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )
    
#Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ),
                                  
                                  ( 'num_transformer', NumericalTransformer() ),
                                  
                                  ('imputer', SimpleImputer(strategy = 'median') ),
                                  
                                  ( 'std_scaler', StandardScaler() ) ] )

#Combining numerical and categorical piepline into one full big pipeline horizontally 
#using FeatureUnion
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), 
                                                  
                                                  ( 'numerical_pipeline', numerical_pipeline ) ] )


# In[10]:


# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# #Leave it as a dataframe becuase our pipeline is called on a 
# #pandas dataframe to extract the appropriate columns, remember?
# #X = data.drop('price', axis = 1)
# #You can covert the target variable to numpy 
# #y = data['price'].values 

# X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )

# #The full pipeline as a step in another pipeline with an estimator as the final step
# full_pipeline_m = Pipeline( steps = [ ( 'full_pipeline', full_pipeline),
                                  
#                                   ( 'model', LinearRegression() ) ] )

# #Can call fit on it just like any other pipeline
# full_pipeline_m.fit( X_train, y_train )

# #Can predict with it like any other pipeline
# y_pred = full_pipeline_m.predict( X_test ) 


# In[ ]:




