import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_model_params(models, path = '/content/Stark_ML/Results'):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    
    params = {}
    for model in models:
        with open(path + f'/{model}' + '_optimal_parameters', 'r') as fp:
            params[f'{i}'] = json.load(fp)
    
    return params
    
def create_models_dict(models, params):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    if not isinstance(params, dict):
        raise TypeError(f"'params' parameter must be 'dict', not {type(params)}")
        
    models_dict = {}
    
    for model in models:
        models_dict[f'{model}'] = name_to_model(model)(params[f'model'])
        
    return models_dict


def plot_model_comparison(results, figsize = (15, 8), y = 'mse'):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = results, y = y, x="model", ax = ax)
    ax.set_xlabel("", size=40)
    ax.set_ylabel("MSE", size=20)
    ax.set_title("Estimators vs MSE", size=30)
    plt.show()