from flask import Flask, request, jsonify
from itertools import compress

import numpy as np
import pandas as pd

from Stark_ML.utils.encoding import *
from Stark_ML.utils.comms    import *
from Stark_ML.utils.predict  import *

predictor = Predictor()

app = Flask(__name__)

@app.route('/calculate', methods = ['POST'])
def Stark_predict():
    # Extract the value from the URL
    params = request.args
    input_type = params.get('input')                          #query or parse                       <mandatory>
    elements   = params.get('elements')                       #str: NIST-like                       <optional> if input=='query'
    lower      = params.get('lowwl')                          #float                                <optional> if input=='query'
    upper      = params.get('upwl')                           #float                                <optional> if input=='query'
    T_mode     = params.get('T')                              #str 'oneT' or 'multiT'               <mandatory>
    only_T     = params.get('onlyT')                          #float                                <optional> if T=='oneT'
    low_T      = params.get('lowT')                           #float                                <optional> if T=='multiT'
    high_T     = params.get('upT')                            #float                                <optional> if T=='multiT'
    T_step     = params.get('dT')                             #float                                <optional> always
    target     = params.get('output')                         #str 'both' or 'widths' or 'shifts'   <mandatory>
    symbol_out = params.get('out_sym')                        #boolean                              <mandatory>
    wavel_out  = params.get('out_wl')                         #boolean                              <mandatory>
    temp_out   = params.get('out_temp')                       #boolean                              <mandatory>
    charge_out = params.get('out_chrg')                       #boolean                              <mandatory>
    
    save_for_manual_check = True
    
    def _handle_query(spectra: str,
                    lower: float,
                    upper: float):
        elements, ionizations = convert_species_request(spectra)

        connection = connect_to_DB(username = '',
                                  password = '')
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
            dtypes = data.dtypes.to_dict()
            for index, row in data.iterrows():
                data.at[index, 'T'] = only_T
            data = data.astype(dtypes)
            return data

        if T_mode == 'multiT':
            dtypes = data.dtypes.to_dict()
            Ts = np.arange(low_T, high_T + 1, T_step)
            for index, row in data.iterrows():
                data.at[index, 'T'] = low_T
                for T in Ts:
                    if T == low_T:
                        continue
                    row['T'] = T
                    data = pd.concat([data, row.to_frame().T], ignore_index=True)
            data = data.astype(dtypes)
            return data
        
        
    
    DB_df = _handle_query(elements, lower, upper)
    
    data_i = pd.read_excel(Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/Stark_data.xlsx',
                           sheet_name='Ions',
                           usecols='A:BQ',
                           nrows = 2
                       )
    request_df = split_OK_check(DB_to_StarkML(DB_df, data_i), save_manual_check = save_for_manual_check, save_txts = False)
    
    
    
    request_df.insert(request_df.columns.get_loc('E upper')+1, 'Gap to ion', 0)
    request_df['Gap to ion'] = gap_to_ion(request_df, 'E upper')
    request_df = request_df
    
    #request_df = _add_temperature(request_df, T_mode)
    
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
        
    
    return jsonify(results.to_json(orient = 'records'))


@app.route('/count_lines', methods = ['POST'])
def count_lines():
    results = 'bbb'
    return jsonify(results)
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)