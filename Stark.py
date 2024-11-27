from flask import Flask, request, jsonify, make_response
from itertools import compress

import json
import io
import numpy as np
import pandas as pd
import datetime as dt

from Stark_ML.utils.encoding import *
from Stark_ML.utils.comms    import *
from Stark_ML.utils.predict  import *

predictor = Predictor()
#with open('Stark_ML/credentials.json', 'r') as file:
#    creds = json.load(file)

app = Flask(__name__, static_url_path='/', static_folder="C:/Users/Alex/Documents/GitHub/spmodel-webui-starkml/webapp/public")


@app.route('/cgi-bin/starkml.jar', methods = ['POST'])
def Stark_predict():
    # Extract the value from the URL
    params = request.args
    
    input_type = request.form.get('input', None)                          #query or parse                       <mandatory>
    elements   = request.form.get('elements', None)                       #str: NIST-like                       <optional> if input=='query'
    lower      = request.form.get('lowwl', None)                          #float                                <optional> if input=='query'
    upper      = request.form.get('upwl', None)                           #float                                <optional> if input=='query'
    T_mode     = request.form.get('T', None)                              #str 'oneT' or 'multiT'               <mandatory>
    only_T     = request.form.get('onlyT', None)                          #float                                <optional> if T=='oneT'
    low_T      = request.form.get('lowT', None)                           #float                                <optional> if T=='multiT'
    high_T     = request.form.get('upT', None)                            #float                                <optional> if T=='multiT'
    T_step     = request.form.get('dT', None)                             #float                                <optional> always
    target     = request.form.get('output', None)                         #str 'both' or 'widths' or 'shifts'   <mandatory>
    symbol_out = request.form.get('out_sym', None)                        #boolean                              <mandatory>
    wavel_out  = request.form.get('out_wl', None)                         #boolean                              <mandatory>
    temp_out   = request.form.get('out_temp', None)                       #boolean                              <mandatory>
    charge_out = request.form.get('out_chrg', None)                       #boolean                              <mandatory>
    
    save_for_manual_check = True
    
    
    def _add_temperature(data: pd.DataFrame,
                        T_mode: str,
                        only_T = None,
                        low_T = None,
                        high_T = None,
                        T_step = None):
        if T_mode == 'oneT':
            dtypes = data.dtypes.to_dict()
            for index, row in data.iterrows():
                data.at[index, 'T'] = float(only_T)
            data = data.astype(dtypes)
            data['T'] = data['T'].astype(float)
            return data

        if T_mode == 'multiT':
            dtypes = data.dtypes.to_dict()
            low_T, high_T = min(float(low_T), float(high_T)), max(float(low_T), float(high_T))
            Ts = np.arange(low_T, high_T + 1, abs(float(T_step)))
            for index, row in data.iterrows():
                data.at[index, 'T'] = low_T
                for T in Ts:
                    if T == low_T:
                        continue
                    row['T'] = T
                    data = pd.concat([data, row.to_frame().T], ignore_index=True)
            data = data.astype(dtypes)
            data['T'] = data['T'].astype(float)
            return data
        
        
    def _send_response(data, for_check):
        if request.accept_mimetypes['application/json']:
            ret = data.to_dict(orient = 'list')
            if for_check is not None:
                ret['unparsed'] = for_check.fillna(0).to_dict(orient = 'list')
            return jsonify(ret)
        elif request.accept_mimetypes['text/plain']:
            response = make_response(data.to_csv(sep = '\t', index = False))
            response.headers['Content-Type']        = 'text/plain'
            response.headers['Content-Disposition'] = 'attachment; filename = "prediction.txt"'
            return response
    
    
    if input_type == 'query':
        
        try:
            request_df, lines_for_check = get_lines_from_DB(elements, lower, upper, save_for_manual_check = save_for_manual_check)
        except UserDefinedError as e:
            return jsonify({'error': str(e)})
    elif input_type == 'parse':
        if 'upload' not in request.files:
            return jsonify({'error': 'No file'})
        file = request.files['upload']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        request_df = pd.read_csv(file, compression = None)
        lines_for_check = None
            
#    print(f'lines for check: {lines_for_check}')
    if request_df.empty:
        return jsonify({'error': 'No lines could be encoded properly. Please, check them manually in the file below', 'unparsed':lines_for_check.fillna(0).to_dict(orient = 'list')})
    
    
    #Check if request_df length exceeds 5000 rows limit
    n_temperatures = 1 if T_mode == 'oneT' else abs(float(high_T) - float(low_T))//abs(float(T_step)) + 1
    if request_df.shape[0] > 5000 or request_df.shape[0]*n_temperatures > 5000:
        return jsonify({'error': '5000 rows at once is the limit, sorry'})
    
        
    request_df.insert(request_df.columns.get_loc('E upper')+1, 'Gap to ion', 0)
    request_df['Gap to ion'] = gap_to_ion(request_df, 'E upper')
    request_df = _add_temperature(request_df, T_mode, only_T, low_T, high_T, T_step)
    
    
#    If elements == all, drop high ionization degrees to unparsed
#    If elements are specified with too high ionization degrees -> raise error
    if request_df['Gap to ion'].isna().any():
        if elements == '':
            lines_for_check = pd.concat(
                [lines_for_check,
                request_df[request_df['Gap to ion'].isna() == True]],
                axis = 0,
                ignore_index = True
                )
            request_df = request_df[~request_df['Gap to ion'].isna() == True].reset_index(drop = True)
        else:
            return jsonify({'error': f'Cannot get predictions for element {request_df[request_df["Gap to ion"].isna() == True].iloc[0]["Element"]} with charge {request_df[request_df["Gap to ion"].isna() == True].iloc[0]["Charge"]}'})
    
    
    request_df = request_df.sort_values(by = ['Wavelength', 'T'], ignore_index = True)
    
    
    
    #start_time = dt.datetime.now()
    if target == 'widths':
        preds = predictor.predict_width(request_df)
        preds = pd.Series(preds, name = 'w (A)')
    if target == 'shifts':
        preds = predictor.predict_shift(request_df)[:, 1]
        preds = pd.Series(preds, name = 'd (A)')
    if target == 'both':
        preds = predictor.predict_shift(request_df)
        preds = pd.DataFrame(preds, columns = ['w (A)', 'd (A)'])
    #print(f'Prediction of both parameters for {request_df.shape[0]} entries takes {dt.datetime.now() - start_time}')
        
    
    columns = ['Element', 'Charge', 'Wavelength', 'T', 'w (A)', 'd (A)']
    
    results = pd.DataFrame(columns = list(compress(columns, [symbol_out, charge_out, wavel_out, temp_out,
                                                         True if (target == 'widths') | (target == 'both') else False,
                                                         True if (target == 'shifts') | (target == 'both') else False])))
    results = pd.concat(
            [
            request_df[list(compress(columns, [symbol_out, charge_out, wavel_out, temp_out]))],
            preds,
            ],
        axis = 1
        )
       
    return _send_response(results, lines_for_check)


#@app.route('/cgi-bin/count_lines.rb', methods = ['POST'])
@app.route('/cgi-bin/count_rows.rb', methods = ['POST'])
#def count_lines():
def count_rows():
    
    input_type = request.form.get('input', None)                          #query or parse                       <mandatory>
    elements   = request.form.get('elements', None)                       #str: NIST-like                       <optional> if input=='query'
    lower      = request.form.get('lowwl', None)                          #float                                <optional> if input=='query'
    upper      = request.form.get('upwl', None)                           #float                                <optional> if input=='query'
    T_mode     = request.form.get('T', None)                              #str 'oneT' or 'multiT'               <mandatory>
    only_T     = request.form.get('onlyT', None)                          #float                                <optional> if T=='oneT'
    low_T      = request.form.get('lowT', None)                           #float                                <optional> if T=='multiT'
    high_T     = request.form.get('upT', None)                            #float                                <optional> if T=='multiT'
    T_step     = request.form.get('dT', None)                             #float                                <optional> always
    
    if input_type == 'parse':
        if 'upload' not in request.files:
                return jsonify({'error': 'No file'})
        file = request.files['upload']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        request_df = pd.read_csv(file, compression = None)
        rows_count = request_df.shape[0]
    elif input_type == 'query':
        rows_count = get_lines_from_DB(elements, lower, upper, count_mode = True)
    
    
    if T_mode == 'oneT':
        return jsonify({'count':f'{rows_count}'})
    elif T_mode == 'multiT':
        n_temperatures = abs(float(high_T) - float(low_T))//abs(float(T_step)) + 1
        return jsonify({'count':f'{int(rows_count*n_temperatures)}'})
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)