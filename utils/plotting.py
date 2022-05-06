import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def name_to_model(model):
    if "KNN" in model:
        from Stark_ML.models.base_models import KNN
        return KNN
        
    elif "RF" in model:
        from Stark_ML.models.base_models import RandomForest
        return RandomForest
        
    elif "XGB" in model:
        from Stark_ML.models.base_models import XGBoost
        return XGBoost
        
    elif "GB" in model:
        from Stark_ML.models.base_models import Gradient_Boosting
        return Gradient_Boosting
    
    elif "TabNet" in model:
        from Stark_ML.models.tabnet import TabNet
        return TabNet
        
    else:
        raise NotImplementedError(f'Model {model} has not been implemented yet')
        
        
def get_model_params(models, path = '/content/Stark_ML/Results'):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    
    params = {}
    for model in models:
        with open(path + f'/{model}' + '_optimal_parameters', 'r') as fp:
            params[f'{model}'] = json.load(fp)
    
    return params
    
def create_models_dict(models, params = None):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    if params == None:
        params = get_model_params(models)
    if not isinstance(params, dict):
        raise TypeError(f"'params' parameter must be 'dict', not {type(params)}")
        
    models_dict = {}
    
    for model in models:
        if f'{model}' not in params:
            params['f{model}'] = get_model_params([f'{model}'])['f{model}']
        models_dict[f'{model}'] = name_to_model(model)(params[f'{model}'])
        
    return models_dict


def plot_model_comparison(results, figsize = (15, 8), y = 'mse'):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = results, y = y, x="model", ax = ax)
    ax.set_xlabel("", size=40)
    ax.set_ylabel("MSE", size=20)
    ax.set_title("Estimators vs MSE", size=30)
    plt.show()
    
    
def plot_model_prediction(models, X_train, y_train, X_test, y_test, X_elem = None, y_elem = None, label_elem = None):
    '''
    Takes models (dict) and data as input, returns predictions and plots
    '''
    if X_elem is None and y_elem is None:
        grid_h = 1
    elif X_elem is not None and y_elem is not None:
        grid_h = 2
    else:
        raise ValueError(f"'X_elem' and 'y_elem' must be both either 'None' or not.")
        
    predictions = {}
    predictions_elem = {}
    R2 = {}
    R2_elem = {}
    for name, model in models.items():
        print(f"Getting {name} predictions")
        if 'TabNet' in name:
            model.fit(X_train, y_train, X_test, y_test)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        R2[name] = r2_score(y_test, y_pred)
        
        if grid_h == 2:
            y_pred = model.predict(X_elem)
            predictions_elem[name] = y_pred
            R2_elem[name] = r2_score(y_elem, y_pred)
        
        
    i = 0
    fig, ax = plt.subplots(grid_h, len(models), figsize = (5*len(models), 4*grid_h))
    for name, model in models.items():
        if grid_h == 1:
            ax[i].plot(y_test, predictions[name], 'r.')
            ax[i].plot([0, np.amax(y_test)], [0, np.amax(y_test)], color = 'b', ls = '--')
            ax[i].set_title(f'{name}')
            ax[i].text(x = 0, y = 1, s = f'$R^2$ = {R2[name]:.4f}', transform = ax[i].transAxes)
        else:
            ax[0, i].plot(y_test, predictions[name], 'r.')
            ax[0, i].plot([0, np.amax(y_test)], [0, np.amax(y_test)], color = 'b', ls = '--')
            ax[0, i].set_title(f'{name}')
            ax[0, i].text(x = 0, y = 1, s = f'$R^2$ = {R2[name]:.4f}', transform = ax[0, i].transAxes)
            
            #ax[1, i].plot(y_elem, predictions_elem[name], 'r.')
            sns.scatterplot(y_elem, predictions_elem[name], ax = ax[1, i])
            ax[1, i].plot([0, np.amax(y_elem)], [0, np.amax(y_elem)], color = 'b', ls = '--')
            ax[1, i].text(x = 0, y = 1, s = f'$R^2$ = {R2_elem[name]:.4f}',  transform = ax[1, i].transAxes)
        i += 1
    plt.show()

    return predictions, predictions_elem, fig, ax