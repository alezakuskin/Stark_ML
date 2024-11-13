from flask import Flask, request, jsonify, make_response
from itertools import compress

import json
import numpy as np
import pandas as pd

from Stark_ML.utils.encoding import *
from Stark_ML.utils.comms    import *
from Stark_ML.utils.predict  import *

predictor = Predictor()
with open('Stark_ML/credentials.json', 'r') as file:
    creds = json.load(file)

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
    
    def _handle_query(spectra: str,
                    lower: float,
                    upper: float):
        elements, ionizations = convert_species_request(spectra)

        connection = connect_to_DB(username = creds['username'],
                                  password = creds['password'])
        cur = connection.cursor()
        DB_df = None
        for i in range(len(elements)):
            el = elements[i]
            ion = ionizations[i]
            if ion != 'All':
                query = f'''
                SELECT *
                FROM mytestview2
                WHERE airwl >= ?
                AND airwl <= ?
                AND el_name = ?
                AND ion_stage in {f"({', '.join(map(str, ion))})"}
                '''
            else:
                query = f'''
                SELECT *
                FROM mytestview2
                WHERE airwl >= ?
                AND airwl <= ?
                AND el_name = ?
                '''
            cur.execute(query, (lower, upper, el))
            column_names = [desc[0] for desc in cur.description]
            req_results = cur.fetchall()
            req_results = pd.DataFrame(req_results, columns=column_names)
            if not req_results.empty:
                if DB_df is None:
                    DB_df = req_results
                else:
                    DB_df = pd.concat([DB_df, req_results], ignore_index=True)
        cur.close()
        connection.close()
        return DB_df

    def _add_temperature(data: pd.DataFrame,
                        T_mode: str):
        if T_mode == 'oneT':
            #only_T = float(only_T)
            dtypes = data.dtypes.to_dict()
            for index, row in data.iterrows():
                data.at[index, 'T'] = float(only_T)
            data = data.astype(dtypes)
            return data

        if T_mode == 'multiT':
            dtypes = data.dtypes.to_dict()
            Ts = np.arange(float(low_T), float(high_T) + 1, float(T_step))
            for index, row in data.iterrows():
                data.at[index, 'T'] = float(low_T)
                for T in Ts:
                    if T == float(low_T):
                        continue
                    row['T'] = T
                    data = pd.concat([data, row.to_frame().T], ignore_index=True)
            data = data.astype(dtypes)
            return data
        
        
    def _send_response(data):
        if request.accept_mimetypes['application/json']:
            return jsonify(data.to_dict(orient = 'list'))
        elif request.accept_mimetypes['text/plain']:
            response = make_response(data.to_csv(sep = '\t', index = False))
            response.headers['Content-Type']        = 'text/plain'
            response.headers['Content-Disposition'] = 'attachment; filename = "prediction.txt"'
            return response
    
        
    def _get_lines_from_DB():
        DB_df = _handle_query(elements, lower, upper)
        data_i = pd.read_excel(Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/Stark_data.xlsx',
                               sheet_name='Ions',
                               usecols='A:BQ',
                               nrows = 2
                           )
        try:
            request_df, lines_for_check = split_OK_check(DB_to_StarkML(DB_df, data_i), save_manual_check = save_for_manual_check, save_txts = False)
            return request_df, lines_for_check
        except UserDefinedError as e:
            raise UserDefinedError(e)
            
        
    if input_type == 'query':
        try:
            request_df, lines_for_check = _get_lines_from_DB()
        except UserDefinedError as e:
            return jsonify({'error': str(e)})
    elif input_type == 'parse':
        if 'file' not in request.files:
            return jsonify({'error': 'No file'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        #try:
        request_df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        #except:
            
    
    request_df.insert(request_df.columns.get_loc('E upper')+1, 'Gap to ion', 0)
    request_df['Gap to ion'] = gap_to_ion(request_df, 'E upper')
    request_df = _add_temperature(request_df, T_mode)
    
    request_df = request_df.sort_values(by = ['Wavelength', 'T'], ignore_index = True)
    
    if target == 'widths':
        preds = predictor.predict_width(request_df)
        preds = pd.Series(preds, name = 'w (A)')
    if target == 'shifts':
        preds = predictor.predict_shift(request_df)[:, 1]
        preds = pd.Series(preds, name = 'd (A)')
    if target == 'both':
        preds = predictor.predict_shift(request_df)
        preds = pd.DataFrame(preds, columns = ['w (A)', 'd (A)'])
        
    
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
       
    #_send_response(results)
    return _send_response(results)


@app.route('/count_lines', methods = ['POST'])
def count_lines():
    results = 'bbb'
    return jsonify(results)
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)