from flask import Flask, request, jsonify
from itertools import compress

import numpy as np
import pandas as pd

from Stark_ML.utils.encoding import *
from Stark_ML.utils.comms    import *
from Stark_ML.utils.predict  import *

app = Flask(__name__)

@app.route('/calculate', methods = 'POST')
def Stark_predict():
    # Extract the value from the URL
    params = request.args
    input_type = params.get('input')
    elements   = params.get('elements')
    
    
    
    
    
	#spectra    = params.get('spectra')
    #lower    = params.get('lower')
    #upper    = params.get('upper')
    #target   = params.get('target')
    #save_for_manual_check = params.get('save_for_manual_check')
    #filename = params.get('filename')
    #Temperature_mode = params.get('Temperature_mode')
    #Low_T    = params.get('Low_T')
    #High_T   = params.get('High_T')
    #T_step   = params.get('T_step')
    
    
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)








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






