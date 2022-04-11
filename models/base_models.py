#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost


# In[ ]:


class BaseModel:

    def __init__(self, params):
        self.predictions = None

        # Save all hyperparameters which are optimized
        self.params = params

        # Model has to be set by the concrete model
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        self.predictions = self.model.predict(X)
        return self.predictions

    def save_model_and_predictions(self, y_true, filename_extension=""):
        self.save_predictions(y_true, filename_extension)
        self.save_model(filename_extension)

    def clone(self):
        return self.__class__(self.params)

    @classmethod
    def define_trial_parameters(cls, trial):
        raise NotImplementedError("This method has to be implemented by the sub class")

    '''
        Private functions
    '''

    def save_model(self, filename_extension=""):
        print('lll')

    def save_predictions(self, y_true, filename_extension=""):
       print('kkk')


class KNN(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = KNeighborsRegressor(n_neighbors = params['n_neighbors'],
                                         leaf_size = params['leaf_size'],
                                         p = params['p'],
                                         weights = params['weights'],
                                         algorithm = params['algorithm'],
                                         n_jobs=-1)

        self.params = params
        
    def fit(self, X, y, X_val=None, y_val=None):
        
        return super().fit(X, y, X_val, y_val)

    @classmethod
    def define_trial_parameters(cls, trial, params):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", params['n_neighbors'][0], params['n_neighbors'][1]),
            'weights' : trial.suggest_categorical('weights', params['weights']),
            'algorithm' : trial.suggest_categorical('algorithm', params['algorithm']),
            'leaf_size' : trial.suggest_int("leaf_size", params['leaf_size'][0], params['leaf_size'][1]),
            'p' : trial.suggest_float('p', params['p'][0], params['p'][1])
        }
        return params


class RandomForest(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = RandomForestRegressor(n_estimators = params['n_estimators'],
                                           max_depth = params['max_depth'],
                                           min_samples_split = params['min_samples_split'],
                                           min_samples_leaf = params['min_samples_leaf'],
                                           n_jobs = -1)
        
        self.params = params

    @classmethod
    def define_trial_parameters(cls, trial, params):
        params = {
            'n_estimators': trial.suggest_int('n_estimatoprs', params['n_estimators'][0], params['n_estimators'][1], log = True),
            'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False),
            'min_samples_split': trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1], log = False),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1], log = False)
        }
        return params



class Gradient_Boosting(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = GradientBoostingRegressor(
            learning_rate = params['learning_rate'],
            min_samples_split = params['min_samples_split'],
            min_samples_leaf = params['min_samples_leaf'],
            max_depth = params['max_depth']
        )

        self.params = params
    
    @classmethod
    def define_trial_parameters(cls, trial, params):
        params = {
            'learning_rate' : trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1]),
            'min_samples_split': trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1], log = False),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1], log = False),
            'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False)
        }
        return params


class XGBoost(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = xgboost.XGBRegressor(
            learning_rate = params['learning_rate'],
            min_child_weight = params['min_chils_weight'],
            max_depth = params['max_depth']
        )

        self.params = params

    @classmethod
    def define_trial_parameters(cls, trial, params):
        params = {
            'min_child_weight': trial.suggest_int('min_child_weight', params['min_child_weight'][0], params['min_child_weight'][1], log = False),
            'max_depth' : trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False),
            'learning_rate' : trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1])
        }

