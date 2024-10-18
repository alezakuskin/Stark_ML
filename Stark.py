from flask import Flask, request, jsonify
from itertools import compress
from urllib import request, parse

import pandas as pd
import numpy as np
import xgboost
import catboost
import roman
import joblib

from Stark_ML.utils.terms import *


app = Flask(__name__)

@app.route('/')
def Stark_predict():
    # Extract the value from the URL
	params = request.args
	spectra  = params.get('spectra')
    lower    = params.get('lower')
    upper    = params.get('upper')
    target   = params.get('target')
    save_for_manual_check = params.get('save_for_manual_check')
    filename = params.get('filename')
    Temperature_mode = params.get('Temperature_mode')
    Low_T    = params.get('Low_T')
    High_T   = params.get('High_T')
    T_step   = params.get('T_step')
    
    nist_params = { # error if not commented and equals 0
        'spectra': spectra,
        'limits_type': 0,
        'low_w': lower,
        'upp_w': upper,
        'unit': 1,
        'de': 0,
        'I_scale_type': 1,
        'format': 3,
        'line_out': 0,
        'en_unit': 0,
        'output': 0,
        #'bibrefs': 1,
        'page_size': 15,
        'show_obs_wl': 1,
        'show_calc_wl': 1,
        #'unc_out': 0,
        'order_out': 0,
        'max_low_enrg': '',
        'show_av': 2,
        'max_upp_enrg': '',
        'tsb_value': 0,
        'min_str': '',
        #'A_out': 0,
        #'intens_out': 'off',
        'max_str': '',
        'allowed_out': 1,
        'forbid_out': 1,
        'min_accur': '',
        'min_intens': '',
        'conf_out': 'on',
        'term_out': 'on',
        'enrg_out': 'on',
        'J_out': 'on',
        #'g_out': 'on',
        #'remove_js': 'on',
        #'no_spaces': 'on',
        #'show_diff_obs_calc': 0,
        #'show_wn': 1,
        #'f_out': 'off',
        #'S_out': 'off',
        #'loggf_out': 'off',
        'submit': 'Retrieve Data',
    }

    url = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?'
    data = parse.urlencode(nist_params)
    req =  request.Request(url+data)
    with request.urlopen(req) as resp:
        df = pd.read_csv(resp, sep='\t')
    if 'sp_num' in list(df.columns):
        df = df.drop(df.loc[df['sp_num'] == 'sp_num'].index)

    data_i = pd.read_excel(Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/Stark_data.xlsx',
                           sheet_name='Ions',
                           usecols='A:BQ',
                           nrows = 2
                       )
    request_df = split_OK_check(NIST_to_StarkML(df, data_i, spectra), save_manual_check = save_for_manual_check)
    
    filename = 'Stark_ML/' + filename

    try:
        data_predictions = pd.read_csv(filename,
                                       index_col = 0
                                       )
    except:
        data_predictions = pd.read_csv(filename[9:],
                                        index_col = 0
                                        )
                                        
    #Data preprocessing
    data_predictions.insert(data_predictions.columns.get_loc('E upper')+1, 'Gap to ion', 0)
    data_predictions['Gap to ion'] = gap_to_ion(data_predictions, 'E upper')
    data_predictions = data_predictions

    if Temperature_mode == 'single':
        print('here')
        dtypes = data_predictions.dtypes.to_dict()
        for index, row in data_predictions.iterrows():
            data_predictions.at[index, 'T'] = Low_T
        data_predictions = data_predictions.astype(dtypes)

    if Temperature_mode == 'range':
        dtypes = data_predictions.dtypes.to_dict()
        Ts = np.arange(Low_T, High_T + 1, T_step)
        for index, row in data_predictions.iterrows():
            data_predictions.at[index, 'T'] = Low_T
            for T in Ts:
                if T == Low_T:
                    continue
                row['T'] = T
                data_predictions = pd.concat([data_predictions, row.to_frame().T], ignore_index=True)
        data_predictions = data_predictions.astype(dtypes)
    data_predictions = data_predictions.sort_values(['Wavelength', 'T']).reset_index(drop = True)
        
    #Get predictions
    if target == 'broadening':
        preds = predict_width(data_predictions.drop(columns=['Element', 'Wavelength', 'Z number', 'w (A)', 'd (A)']))
        preds = pd.Series(preds, name = 'w (A)')
    if target == 'shift':
        preds = predict_shift(data_predictions.drop(columns=['Element', 'Wavelength', 'Z number', 'w (A)', 'd (A)']))[:, 1]
        preds = pd.Series(preds, name = 'd (A)')
    if target == 'both':
        preds = predict_shift(data_predictions.drop(columns=['Element', 'Wavelength', 'Z number', 'w (A)', 'd (A)']))
        preds = pd.DataFrame(preds, columns = ['w (A)', 'd (A)'])
        
        
    #building output file
    columns = ['Element', 'Charge', 'Wavelength', 'T', 'w (A)', 'd (A)']
    #@markdown

    #@markdown ###Select additional transition parameters you would like to include in output file
    Element_symbol = True  #@param {type: 'boolean'}
    Wavelength     = True  #@param {type: 'boolean'}
    Temperature    = True  #@param {type: 'boolean'}
    Charge         = True #@param {type: 'boolean'}

    results = pd.DataFrame(columns = list(compress(columns, [Element_symbol, Charge, Wavelength, Temperature,
                                                             True if (target == 'broadening') | (target == 'both') else False,
                                                            True if (target == 'shift') | (target == 'both') else False])))
    results = pd.concat(
            [
            data_predictions[list(compress(columns, [Element_symbol, Charge, Wavelength, Temperature]))],
            preds,
            ],
        axis = 1
        )
    results.to_csv(f'PREDICTED_{filename[9:-4]}.csv', index = False)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)








def predict_width(data_for_prediction):
    '''
    Get predicted Stark broadening parameters for input lines
    
    Parameters
    ----------
    data_for_prediction : pd.DataFrame, dataframe with any number of rows,
        all values of input features filled in; without "Element", "Wavelength",
        "Z number", "w (A)", "d (A)" columns.
    
    Returns
    ----------
    numpy.ndarray
        A one-dimentional array with predicted values of broadening parameters in \u212B
    '''
    #Importing pretrained models
    model1 = xgboost.XGBRegressor()
    model1.load_model('Stark_ML/XGB_A+I_Eraw_Raw_No.json')

    model2 = xgboost.XGBRegressor()
    model2.load_model('Stark_ML/XGB_A+I_Enorm_Aug_No.json')

    model3 = catboost.CatBoostRegressor()
    model3.load_model('Stark_ML/CatBoost_A+I_Enorm_Raw_No.json')

    model4 = joblib.load('Stark_ML/LightGBM_A+I_Eraw_Raw_No.pkl')

    model5 = joblib.load('Stark_ML/LightGBM_A+I_Enorm_Raw_Scaler.pkl')

    #Loading Standard Scaler
    scaler = joblib.load('Stark_ML/scaler_width.pkl')
    
    #Getting predictions
    epsilon = 1e-3
    pred1 = model1.predict(data_for_prediction)
    pred2 = model2.predict(data_for_prediction)
    pred3 = model3.predict(data_for_prediction)
    pred4 = model4.predict(data_for_prediction)
    pred5 = model5.predict(scaler.transform(data_for_prediction))
    preds = (pred1 + pred2 + pred3 + pred4 + pred5)/5
    preds = (np.exp(preds) - 1) * epsilon
    
    return(preds)

def predict_shift(data_for_prediction):
    '''
    Get predicted Stark shift parameters for input lines
    
    Parameters
    ----------
    data_for_prediction : pd.DataFrame, dataframe with any number of rows,
        all values of input features filled in; without "Element", "Wavelength",
        "Z number", "w (A)", "d (A)" columns.
    
    Returns
    ----------
    numpy.ndarray
        A two-dimentional array with predicted values of both broadening (1-st column)
        and shift (2nd column) parameters in \u212B
    '''
    #Importing pretrained models
    model = joblib.load('Stark_ML/RF_Both_Eraw_Aug_No.pkl')

    #Get broadening predictions first
    widths = predict_width(data_for_prediction)
    
    #Adjust input data
    data_for_prediction['w (A)'] = widths
    data_for_prediction = data_for_prediction[model.model.feature_names_in_]
    
    #Get shift predictions
    preds = model.predict(data_for_prediction)
    
    return(np.column_stack((widths, preds)))






