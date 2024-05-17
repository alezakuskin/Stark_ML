import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Stark_ML
from sklearn.metrics import r2_score, mean_squared_error
from tqdm.notebook import tqdm


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
    
    elif "LightGBM" in model:
        from Stark_ML.models.base_models import LightGBM
        return LightGBM    
    
    elif "GB" in model:
        from Stark_ML.models.base_models import Gradient_Boosting
        return Gradient_Boosting
    
    elif "TabNet" in model:
        from Stark_ML.models.tabnet import TabNet
        return TabNet
    
    elif "CatBoost" in model:
        from Stark_ML.models.base_models import CatBoost
        return CatBoost   
    else:
        raise NotImplementedError(f'Model {model} has not been implemented yet')
        
        
def get_model_params(models, path = os.path.join(Stark_ML.__path__.__dict__['_path'][0], 'Results')):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    
    params = {}
    for model in models:
        with open(path + f'/{model}' + '_optimal_parameters', 'r') as fp:
            params[f'{model}'] = json.load(fp)
    
    return params
    
def create_models_dict(models, params = None, path = os.path.join(Stark_ML.__path__.__dict__['_path'][0], 'Results')):
    if not isinstance(models, list):
        raise TypeError(f"'models' parameter must be 'list', not {type(models)}")
    if params == None:
        params = get_model_params(models, path)
    if not isinstance(params, dict):
        raise TypeError(f"'params' parameter must be 'dict', not {type(params)}")
        
    models_dict = {}
    
    for model in models:
        if f'{model}' not in params:
            params['f{model}'] = get_model_params([f'{model}'], path)['f{model}']
        models_dict[f'{model}'] = name_to_model(model)(params[f'{model}'])
        
    return models_dict


def plot_model_comparison(results, figsize = (15, 8), y = 'mse'):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = results, y = y, x="model", ax = ax)
    ax.set_xlabel("", size=40)
    ax.set_ylabel("RMSE", size=20)
    ax.set_title("Estimators vs RMSE", size=30)
    plt.show()
    
    return fig
    
    
def plot_model_prediction(models, X_train, y_train, X_test, y_test, X_elem = None, y_elem = None, label_elem = None, plot = True):
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
    RMSE = {}
    RMSE_elem = {}
    MRE = {}
    MRE_elem = {}
    
    for name, model in models.items():
        if 'TabNet' in name:
            model.fit(X_train, y_train, X_test, y_test)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        if plot:
            R2[name] = r2_score(y_test, y_pred)
            RMSE[name] = root_mean_squared_error(y_test, y_pred)
            MRE[name] = (np.abs(y_test - y_pred.reshape(y_pred.shape[0])) / y_test).mean()

        if grid_h == 2:
            y_pred = model.predict(X_elem)
            predictions_elem[name] = y_pred.flatten()
            
            if plot:
                R2_elem[name] = r2_score(y_elem, y_pred)
                RMSE_elem[name] = root_mean_squared_error(y_elem, y_pred)
                MRE_elem[name] = (np.abs(y_elem - y_pred.reshape(y_pred.shape[0])) / y_elem).mean()
        
    if plot == True:
        i = 0
        fig, ax = plt.subplots(grid_h, len(models), figsize = (5*len(models), 4*grid_h))
        for name, model in models.items():
            print(f'Plotting {name} predictions')
            if grid_h == 1:
                ax[i].plot(y_test, predictions[name], 'r.')
                ax[i].plot([0, np.amax(y_test)], [0, np.amax(y_test)], color = 'b', ls = '--')
                ax[i].set_title(f'{name}')
                ax[i].text(x = 0, y = 1, s = f'$R^2$ = {R2[name]:.4f}', transform = ax[i].transAxes)
            else:
                ax[0, i].plot(y_test, predictions[name], 'r.')
                ax[0, i].plot([0, np.amax(y_test)], [0, np.amax(y_test)], color = 'b', ls = '--')
                ax[0, i].set_title(f'{name}')
                ax[0, i].text(x = 0, y = 1.02, s = f'$R^2$ = {R2[name]:.4f}  \nRMSE = {RMSE[name]:.4f}',  transform = ax[0, i].transAxes)

                #ax[1, i].plot(y_elem, predictions_elem[name], 'r.')
                sns.scatterplot(x = y_elem, y = predictions_elem[name], ax = ax[1, i], style = label_elem['Element'], hue = label_elem['Element'])
                ax[1, i].plot([0, np.amax(y_elem)], [0, np.amax(y_elem)], color = 'b', ls = '--')
                ax[1, i].text(x = 0, y = 1.01, s = f'$R^2$ = {R2_elem[name]:.4f}    RMSE = {RMSE_elem[name]:.4f}',  transform = ax[1, i].transAxes)
            i += 1
        plt.show()
        return predictions, predictions_elem, fig, ax
    
    else:
        return predictions, predictions_elem
    

def train_ensemble(ensemble):
    glob_path = 'C:\\Users\\Alex\\Documents\\GitHub'
    models_d     = {}
    preds_d      = {}
    preds_elem_d = {}

    for item in tqdm(ensemble):
        if '_A+I_' in item:
            parameter = 'width'
        elif '_Shift_' in item:
            parameter = 'shift'
        elif '_Both_' in item:
            parameter = 'both'
        else:
            raise NameError(f"Parameter for prediction must be specified in model's name: {item}")

        if 'KNN' in item:
            path = glob_path + '\\KNN'
        elif 'RF' in item:
            path = glob_path + '\\RF'
        elif 'XGB' in item:
            path = glob_path + '\\XGB'
        elif 'LightGBM' in item:
            path = glob_path + '\\LightGBM'
        elif 'CatBoost' in item:
            path = glob_path + '\\CatBoost'

        models_d_item = create_models_dict([item], path = path)

        if '_Eraw_' in item:
            normalized_energy = False
        elif '_Enorm' in item:
            normalized_energy = True

        if '_Raw_' in item:
            augmented_train_set = False
        elif '_Aug_' in item:
            augmented_train_set = True

        if '_No' in item:
            apply_scaler = False
        elif '_Scaler' in item:
            apply_scaler = True
        
        X_train, Y_train, X_test, Y_test, X_elem, Y_elem, L_elem, scaler = constr_train_test(parameter,
                                                                                             augmented_train_set,
                                                                                             scaled_target=True,
                                                                                             normalized_energy = normalized_energy,
                                                                                             print_stats = False)
        if apply_scaler:
            preds, preds_elem = plot_model_prediction(models_d_item,
                                                      scaler.transform(X_train), Y_train,
                                                      scaler.transform(X_test), Y_test,
                                                      scaler.transform(X_elem), Y_elem, L_elem, 
                                                      plot = False)
        else:
            preds, preds_elem = plot_model_prediction(models_d_item,
                                                      X_train, Y_train,
                                                      X_test, Y_test,
                                                      X_elem, Y_elem, L_elem, 
                                                      plot = False)
        preds_d = preds_d | preds
        preds_elem_d = preds_elem_d | preds_elem
    
    return preds_d, preds_elem_d  


def constr_train_test(parameter, augmented_train_set, scaled_target, normalized_energy, print_stats = True):
    #Applying 'width' or 'shift' or 'both' selection
    if parameter == 'width':
        X_train, Y_train = data_width_train.copy(), target_width_train.copy()
        X_test, Y_test = data_width_test.copy(), target_width_test.copy()
        X_elem, Y_elem, L_elem = data_width_elements.copy(), target_width_elements.copy(), label_width_elements.copy()
    elif parameter == 'shift':
        X_train, Y_train = data_shift_train, target_shift_train
        X_test, Y_test = data_shift_test, target_shift_test
        X_elem, Y_elem, L_elem = data_shift_elements, target_shift_elements, label_shift_elements
    elif parameter == 'both':
        X_train, Y_train = data_both_train, target_both_train
        X_test, Y_test = data_both_test, target_both_test
        X_elem, Y_elem, L_elem = data_both_elements, target_both_elements, label_both_elements
    else:
        raise NameError('Incorrect parameter name selected')

    #Handling augmentation
    if augmented_train_set:
        factor = 1.05
        X_train_aug, Y_train_aug = X_train.copy(), Y_train.copy()
        for index, row in X_train.iterrows():
            row['T'] = row['T']*factor
            X_train_aug = pd.concat([X_train_aug, row.to_frame().T], ignore_index=True)
            Y_train_aug = pd.concat([Y_train_aug, pd.Series(Y_train.loc[index])], ignore_index=True)

            row['T'] = row['T']/factor**2
            X_train_aug = pd.concat([X_train_aug, row.to_frame().T], ignore_index=True)
            Y_train_aug = pd.concat([Y_train_aug, pd.Series(Y_train.loc[index])], ignore_index=True)
        X_train_aug = X_train_aug.astype(X_train.dtypes.to_dict())    

        X_train, Y_train = X_train_aug, Y_train_aug
        X_test,  Y_test  = X_test, Y_test

    #Shuffling
    X_train = X_train.sample(frac = 1, random_state = 777)
    Y_train = Y_train.sample(frac = 1, random_state = 777)

    #Applying upper boundary to width values
#     if parameter == 'width':
#         X_train, Y_train = X_train.loc[Y_train.loc[Y_train < width_threshold].index], Y_train.loc[Y_train < width_threshold]
#         X_test,  Y_test  = X_test.loc[Y_test.loc[Y_test < width_threshold].index], Y_test.loc[Y_test < width_threshold]
#         X_elem,  Y_elem  = X_elem.loc[Y_elem.loc[Y_elem < width_threshold].index], Y_elem.loc[Y_elem < width_threshold]
#         L_elem = L_elem.loc[Y_elem.loc[Y_elem < width_threshold].index]
    
    #Applying scaling of targets
    if scaled_target:
        Y_train = np.log(1 + Y_train / epsilon)
        Y_test  = np.log(1 + Y_test / epsilon)
        Y_elem  = np.log(1 + Y_elem / epsilon)

    #Normalizing energies
    if normalized_energy:
        X_train['E lower'], X_train['E upper'] = energy_to_fraction(X_train, 'E lower'), energy_to_fraction(X_train, 'E upper')
        X_test['E lower'],  X_test['E upper']  = energy_to_fraction(X_test, 'E lower'), energy_to_fraction(X_test, 'E upper')
        X_elem['E lower'],  X_elem['E upper']  = energy_to_fraction(X_elem, 'E lower'), energy_to_fraction(X_elem, 'E upper')
        X_train['Gap to ion'] = energy_to_fraction(X_train, 'Gap to ion')
        X_test['Gap to ion'] = energy_to_fraction(X_test, 'Gap to ion')
        X_elem['Gap to ion'] = energy_to_fraction(X_elem, 'Gap to ion')

    #Removing unneccesary columns:
    X_train = X_train.drop(columns=['Element', 'Wavelength', 'Z number'])
    X_test  = X_test.drop(columns=['Element', 'Wavelength', 'Z number'])
    X_elem  = X_elem.drop(columns=['Element', 'Wavelength', 'Z number'])

    scaler = StandardScaler()
    scaler.fit(X_train)

    if print_stats:
        print(f'Selected parameter: {parameter} \n')
        if augmented_train_set:
            print(f'Total number of items with known {parameter}: {X_train.shape[0]/3 + X_test.shape[0] + X_elem.shape[0]}')
        else:
            print(f'Total number of items with known {parameter}: {X_train.shape[0] + X_test.shape[0] + X_elem.shape[0]}')

        print(f'Size of training set: {X_train.shape[0]}')
        print(f'Size of ttest set: {X_test.shape[0]}')
        print(f'Size of elements-exclusive test set: {X_elem.shape[0]}')
    
    return(X_train, Y_train, X_test, Y_test, X_elem, Y_elem, L_elem, scaler)