import json
import matplotlib.pyplot as plt
import seaborn as sns

def name_to_model(model):
    if model == "KNN":
        from Stark_ML.models.base_models import KNN
        return KNN
        
    elif model == "RF":
        from Stark_ML.models.base_models import RandomForest
        return RandomForest
        
    elif model == "XGB":
        from Stark_ML.models.base_models import XGBoost
        return XGBoost
        
    elif model == "GB":
        from Stark_ML.models.base_models import Gradient_Boosting
        return Gradient_Boosting
    
    elif model == "TabNet":
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
        print(model)
        if f'{model}' not in params:
            params['f{model}'] = get_model_params([f'{model}'])['f{model}']
        models_dict[f'{model}'] = name_to_model(model)(params[f'model'])
        
    return models_dict


def plot_model_comparison(results, figsize = (15, 8), y = 'mse'):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = results, y = y, x="model", ax = ax)
    ax.set_xlabel("", size=40)
    ax.set_ylabel("MSE", size=20)
    ax.set_title("Estimators vs MSE", size=30)
    plt.show()