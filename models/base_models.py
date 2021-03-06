import random

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost

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
        
        self.model = KNeighborsRegressor(**params, n_jobs = -1 )
        
        self.params = params
        
    def fit(self, X, y, X_val=None, y_val=None):
        
        return super().fit(X, y, X_val, y_val)

    @classmethod
    def define_trial_parameters(cls, trial, params):
    
        params_tunable = {}
        params_out = {}
        for i, val in params.items():
            if isinstance(val, list):
                params_tunable[f'{i}'] = val
            else:
                params_out[f'{i}'] = val

        if 'n_neighbors' in params_tunable:
            params_out[f'n_neighbors'] = trial.suggest_int("n_neighbors", params_tunable['n_neighbors'][0], params_tunable['n_neighbors'][1])
        if 'weights' in params_tunable:
            params_out[f'weights'] = trial.suggest_categorical('weights', params_tunable['weights'])
        if 'algorithm' in params_tunable:
            params_out[f'algorithm'] = trial.suggest_categorical('algorithm', params_tunable['algorithm'])
        if 'leaf_size' in params_tunable:
            params_out[f'leaf_size'] = trial.suggest_int("leaf_size", params_tunable['leaf_size'][0], params_tunable['leaf_size'][1])
        if 'p' in params_tunable:
            params_out[f'p'] = trial.suggest_float('p', params_tunable['p'][0], params_tunable['p'][1])
        
        if 'nfold' in params_out:
            del params_out['nfold']
        if 'squared_metrics' in params_out:
            del params_out['squared_metrics']
        if 'device_name' in params_out:
            del params_out['device_name']
        if 'n_jobs' in params_out:
            del params_out['n_jobs']
        
        return params_out


class RandomForest(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = RandomForestRegressor(**params, n_jobs = -1)
                                           
        self.params = params

    @classmethod
    def define_trial_parameters(cls, trial, params):
        params_tunable = {}
        params_out = {}
        for i, val in params.items():
            if isinstance(val, list):
                params_tunable[f'{i}'] = val
            else:
                params_out[f'{i}'] = val

        if 'n_estimators' in params_tunable:
            params_out[f'n_estimators'] = trial.suggest_int('n_estimatoprs', params['n_estimators'][0], params['n_estimators'][1], log = True)
        if 'max_depth' in params_tunable:
            params_out[f'max_depth'] = trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False)
        if 'min_samples_split' in params_tunable:
            params_out[f'min_samples_split'] = trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1], log = False)
        if 'min_samples_leaf' in params_tunable:
            params_out[f'min_samples_leaf'] = trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1], log = False)
        if 'criterion' in params_tunable:
            params_out[f'criterion'] = trial.suggest_categorical('criterion', params['criterion'])
        
        if 'nfold' in params_out:
            del params_out['nfold']
        if 'squared_metrics' in params_out:
            del params_out['squared_metrics']
        if 'device_name' in params_out:
            del params_out['device_name']
        if 'n_jobs' in params_out:
            del params_out['n_jobs']
        
        return params_out



class Gradient_Boosting(BaseModel):

    def __init__(self, params):
        super().__init__(params)

        self.model = GradientBoostingRegressor(**params)

        self.params = params
    
    @classmethod
    def define_trial_parameters(cls, trial, params):
        params_tunable = {}
        params_out = {}
        for i, val in params.items():
            if isinstance(val, list):
                params_tunable[f'{i}'] = val
            else:
                params_out[f'{i}'] = val
        
        if 'n_estimators' in params_tunable:
            params_out[f'n_estimators'] = trial.suggest_int('n_estimatoprs', params['n_estimators'][0], params['n_estimators'][1], log = True)
        if 'learning_rate' in params_tunable:
            params_out[f'learning_rate'] = trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1], log = False)
        if 'loss' in params_tunable:
            params_out[f'loss'] = trial.suggest_categorical('loss', params['loss'])
        if 'max_depth' in params_tunable:
            params_out[f'max_depth'] = trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False)
        if 'max_features' in params_tunable:
            params_out[f'max_features'] = trial.suggest_int('max_features', params['max_features'][0], params['max_features'][1], log = False)
        if 'min_samples_split' in params_tunable:
            params_out[f'min_samples_split'] = trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1], log = False)
        if 'min_samples_leaf' in params_tunable:
            params_out[f'min_samples_leaf'] = trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1], log = False)
        
        if 'nfold' in params_out:
            del params_out['nfold']
        if 'squared_metrics' in params_out:
            del params_out['squared_metrics']
        if 'device_name' in params_out:
            del params_out['device_name']
        if 'n_jobs' in params_out:
            del params_out['n_jobs']
        
        return params_out


class XGBoost(BaseModel):
    
    def __init__(self, params):
        super().__init__(params)
        
        self.model = xgboost.XGBRegressor(**params, n_jobs = -1)
    
    @classmethod
    def define_trial_parameters(cls, trial, params):
        params_tunable = {}
        params_out = {}
        for i, val in params.items():
            if isinstance(val, list):
                params_tunable[f'{i}'] = val
            else:
                params_out[f'{i}'] = val
        
        if 'n_estimators' in params_tunable:
            params_out[f'n_estimators'] = trial.suggest_int('n_estimatoprs', params['n_estimators'][0], params['n_estimators'][1], log = True)
        if 'max_depth' in params_tunable:
            params_out[f'max_depth'] = trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log = False)
        if 'max_leaves' in params_tunable:
            params_out[f'max_leaves'] = trial.suggest_int('max_leaves', params['max_leaves'][0], params['max_leaves'][1], log = False)
        if 'learning_rate' in params_tunable:
            params_out[f'learning_rate'] = trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1], log = False)
        if 'min_child_weight' in params_tunable:
            params_out[f'min_child_weight'] = trial.suggest_int('min_child_weight', params['min_child_weight'][0], params['min_child_weight'][1], log = False)
        if 'reg_alpha' in params_tunable:
            params_out[f'reg_alpha'] = trial.suggest_float('reg_alpha', params['reg_alpha'][0], params['reg_alpha'][1], log = True)
        if 'reg_lambda' in params_tunable:
            params_out[f'reg_lambda'] = trial.suggest_float('reg_lambda', params['reg_lambda'][0], params['reg_lambda'][1], log = True)
        
        
        if 'nfold' in params_out:
            del params_out['nfold']
        if 'squared_metrics' in params_out:
            del params_out['squared_metrics']
        if 'device_name' in params_out:
            del params_out['device_name']
        if 'n_jobs' in params_out:
            del params_out['n_jobs']
        
        return params_out