import Stark_ML
import pandas as pd
import numpy as np
from tqdm import tqdm
import roman

def term_to_number(term):
  import pandas as pd
  momentum = pd.Series(dtype = 'float64')
  for i, val in enumerate(term):
    if type(val) == type(0):
      momentum.at[i] = val
    if val == 'S':
      momentum.at[i] = 0
    elif val == 'P':
      momentum.at[i] = 1
    elif val == 'D':
      momentum.at[i] = 2
    elif val == 'F':
      momentum.at[i] = 3
    elif val == 'G':
      momentum.at[i] = 4
    elif val == 'H':
      momentum.at[i] = 5
    elif val == 'I':
      momentum.at[i] = 6
    elif val == 'K':
      momentum.at[i] = 7
    else:
      raise NameError(f'Term symbol "{val}" is not specified')
  return momentum
  
  
def uncertainty_to_number(uncertainty_class):
    uncertainty = pd.Series(dtype = 'float64')
    for i, val in uncertainty_class.items():
        if str(val).isnumeric():
            uncertainty.at[i] = val
        elif val == 'A':
            uncertainty.at[i] = 15
        elif val == 'B+':
            uncertainty.at[i] = 23
        elif val == 'B':
            uncertainty.at[i] = 30
        elif val == 'C+':
            uncertainty.at[i] = 40
        elif val == 'C':
            uncertainty.at[i] = 50
        elif val == 'D+':
            uncertainty.at[i] = 75
        elif val == 'D':
            uncertainty.at[i] = 100
        else:
            raise NameError(f'Uncertainty symbol "{val}" is not specified')
    return uncertainty


def gap_to_ion(data, column_name = None, file = Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/E_ion.csv'):
    import pandas as pd
    import numpy as np
    ion_Es = pd.read_csv(file)
    gap = pd.Series()
    for index, val in enumerate(data['Gap to ion']):
        gap.at[index] = float((ion_Es.loc[ion_Es['Element'] == data.loc[index]['Element']][str(data.loc[index]['Charge'])]).iloc[0]) - data.loc[index][column_name]
        if np.isnan(gap.at[index]):
            print(f"Please find and insert to '/Source_files/E_ion.csv' ionization energy value for {data.loc[index]['Element']} with charge {data.loc[index]['Charge']}")
    return gap
    
    
def energy_to_fraction(data, column_name = None, file = Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/E_ion.csv'):
    ion_Es = pd.read_csv(file)
    E_fraction = pd.Series()
    for index, val in data[f'{column_name}'].items():
        E_fraction.at[index] = val / float((ion_Es.loc[ion_Es['Element'] == data.loc[index]['Element']][str(data.loc[index]['Charge'])]).iloc[0])
    return E_fraction
    

def encode_term(term_str):
    '''
    Takes a single string from NIST output, returns list [Multiplicity, Term, Parity]
    '''
    if str(term_str) == 'nan':
#        print(f'The term {term_str} is not in LS coupling')
        return [np.nan, np.nan, np.nan]
    if '[' in term_str or ']' in term_str or term_str == '*' or term_str.isnumeric():
#        print(f'The term {term_str} is not in LS coupling')
        return [np.nan, np.nan, np.nan]
    if '(' in term_str and ')' in term_str:
#        print(f'The term {term_str} is not in LS coupling')
        return [np.nan, np.nan, np.nan]
    if 'K' in term_str:
#        print(f'K terms are currently not supported')
        return [np.nan, np.nan, np.nan]
    if str(term_str).endswith('e'):
#        print(f'Term set equal to energy')
        return [np.nan, np.nan, np.nan]
    
    #Parity
    if term_str.endswith('*'):
        parity = 0
        term_str = term_str.replace('*', '')
    else:
        parity = 1
        
    #Cut irrelevant symbols
    if ' ' in term_str:
        term_str = term_str[1 + term_str.rfind(' '):]
        
    #Multiplicity and term 
    if len(term_str) == 1:
        multiplicity, term = np.nan, np.nan
    elif term_str[1].isnumeric():
        multiplicity = int(term_str[:2])
        try:
            term = term_to_number(term_str[2])[0]
        except:
            term = np.nan
    else:
        multiplicity = int(term_str[0])
        try:
            term = term_to_number(term_str[1])[0]
        except:
            term = np.nan
    return [multiplicity, term, parity]


def encode_J(J_str):
    if 'or' in str(J_str) or ',' in str(J_str):
        return np.nan
    if '?' in str(J_str):
        J_str = J_str[:J_str.find('?')]
    if '/' in str(J_str):
        return float(J_str[:J_str.find('/')]) / float(J_str[1 + J_str.find('/'):])
    else:
        return float(J_str)
    

def single_shell(shell_str):
    if shell_str[-1].isnumeric():
        if shell_str[1].isnumeric():
            key = shell_str[:3]
        else:
            key = shell_str[:2]
        population = int(shell_str[len(key):])
    else:
        key = shell_str
        population = 1
        
    return key, population


def encode_configuration(conf_str):

    max_population = {
    '1s': 2,
    '2s': 2,
    '2p': 6,
    '3s': 2,
    '3p': 6,
    '3d': 10,
    '4s': 2,
    '4p': 6,
    '5s': 2,
    '4d': 10,
    '5p': 6,
    '4f': 14,
    '5d': 10,
    '6s': 2,
    '6p': 6,
    '7s': 2,
    '5f': 14,
    '6d': 10,
    '7p': 6,
    '7d': 10,
    '8s': 2,
    '8p': 6,
    '8d': 10,
    '9s': 2,
    '10s': 2,
    '11s': 2
    }
    
    if not isinstance(conf_str, str):
        return {
            '1s': np.nan
        }
    if conf_str.isnumeric():
        return {
            '1s': np.nan
        }
    
    conf_str = conf_str.replace(' ', '.')
    pop_dict = {}
    shells = [shell for shell in conf_str.split('.') if '(' not in shell and ')' not in shell and len(shell) > 0]
    
    for _, shell in enumerate(shells):
        if '<' in shell and '>' in shell:
            shells[_] = shell[:shell.find('<')]
        if '?' in shell:
            shells[_] = shell[:shell.find('?')]
    
    for key in max_population:
        if key == single_shell(shells[0])[0]:
            break
        else:
            pop_dict[key] = max_population[key]
    
    for shell in shells:
        key, population = single_shell(shell)
    
        if key in pop_dict:
            pop_dict[key] += population
        else:
            pop_dict[key] = population
    
    return pop_dict


def encode_energy(energy_str):
    if str(energy_str).startswith('[') and str(energy_str).endswith(']'):
        return float(energy_str[1:-1])
    if '(' in str(energy_str) and ')' in str(energy_str):
        return float(energy_str[energy_str.find('(')+1 : energy_str.find(')')])
    if '?' in str(energy_str):
        energy_str = energy_str[:energy_str.find('?')]
    if '+x' in str(energy_str):
        return float(energy_str[:energy_str.find('+x')])
    else:
        try:
            energy = float(energy_str)
        except:
            energy = np.nan
        return energy
    

def NIST_to_StarkML(NIST_df, data_template, spectra):
    '''
    fefer
    '''
    req_df = pd.DataFrame(columns = data_template.columns)
    if 'obs_wl_air(nm)' in list(NIST_df.columns):
        wavel_key = 'air'
    else:
        wavel_key = 'vac'
    for index, item in tqdm(NIST_df.iterrows()):
        if 'element' in list(NIST_df.columns):
            req_df.loc[index, 'Element'] = item['element']
        else:
            req_df.loc[index, 'Element'] = spectra[:spectra.find(' ')]
        
        req_df.loc[index, 'Wavelength'] = item[f'obs_wl_{wavel_key}(nm)']
        if req_df.loc[[index], 'Wavelength'].isna()[index]:
            req_df.loc[index, 'Wavelength'] = item[f'ritz_wl_{wavel_key}(nm)']
        
        if  'sp_num' in list(NIST_df.columns):
            req_df.loc[index, 'Charge'] = int(item['sp_num'])-1
        else:
            req_df.loc[index, 'Charge'] = roman.fromRoman(spectra[spectra.find(' ') + 1:]) - 1
        req_df.loc[index, 'E lower'] = encode_energy(item["Ei(cm-1)"])
        req_df.loc[index, 'E upper'] = encode_energy(item["Ek(cm-1)"])

        encode_up   = encode_term(item["term_k"])
        encode_down = encode_term(item["term_i"])
        req_df.loc[index, 'Multiplicity.1'] = encode_up[0]
        req_df.loc[index, 'Multiplicity'] = encode_down[0]
        req_df.loc[index, 'Term.1'] = encode_up[1]
        req_df.loc[index, 'Term'] = encode_down[1]
        req_df.loc[index, 'Parity.1'] = encode_up[2]
        req_df.loc[index, 'Parity'] = encode_down[2]

        req_df.loc[index, 'J'] = encode_J(item["J_i"])
        req_df.loc[index, 'J.1'] = encode_J(item["J_k"])

        encode_up   = encode_configuration(item['conf_k'])
        encode_down = encode_configuration(item['conf_i'])
        for key in encode_up:
            if f'{key}.1' in req_df.columns:
                req_df.loc[index, f'{key}.1'] = encode_up[key]
            else:
                req_df.loc[index, key] = encode_up[key]
        for key in encode_down:
            req_df.loc[index, key] = encode_down[key]

        req_df.loc[index, 'Z number'] = sum([req_df.loc[index, req_df.columns[i]] for i in range(list(req_df.columns).index('Charge'),
                                                                                                 list(req_df.columns).index('Multiplicity')) if str(req_df.loc[index, req_df.columns[i]]) != 'nan'])
    return req_df


def split_OK_check(StarkML_df, save_txts = True, save_manual_check = True):
    StarkML_df = StarkML_df.drop(columns = [col for col in list(StarkML_df.columns)[1 +list(StarkML_df.columns).index('d (A)'):]])
    need_manual_check = pd.DataFrame(columns = StarkML_df.columns)
    
    c0 = StarkML_df['Multiplicity'].isna() == True
    c1 = StarkML_df['Multiplicity.1'].isna() == True
    c2 = StarkML_df['Term'].isna() == True
    c3 = StarkML_df['Term.1'].isna() == True
    c4 = StarkML_df['J'].isna() == True
    c5 = StarkML_df['J.1'].isna() == True
    c6 = StarkML_df['E lower'].isna() == True
    c7 = StarkML_df['E upper'].isna() == True
    cond = c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7
    
    leave_OK = StarkML_df[~cond]
    need_manual_check = StarkML_df[cond]
    
    leave_OK = leave_OK.fillna(0).reset_index(drop = True)
    
    for index, item in leave_OK.iterrows():
        Z1 = sum([leave_OK.loc[index, leave_OK.columns[i]] for i in range(list(StarkML_df.columns).index('Charge'),
                                                                          list(StarkML_df.columns).index('Multiplicity'))])
        Z2 = sum([leave_OK.loc[index, leave_OK.columns[i]] for i in range(list(StarkML_df.columns).index('1s.1'),
                                                                          list(StarkML_df.columns).index('Multiplicity.1'))]) + leave_OK.loc[index, 'Charge']
        if Z1 != Z2:
            need_manual_check = pd.concat([need_manual_check, leave_OK.loc[index].to_frame().T])
            leave_OK = leave_OK.drop([index])
    
    if save_txts:
        if save_manual_check:
            need_manual_check.reset_index(drop = True).to_csv('for_manual_check.txt')
            print(f'{need_manual_check.shape[0]} lines could not be encoded correctly. Please, check them manually in for_manual_check.txt')
        leave_OK.reset_index(drop = True).to_csv('requested_lines.txt')
    
    print(f'{leave_OK.shape[0]} lines were encoded correctly.')
        
    return leave_OK