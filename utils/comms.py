import re
import json
import roman
import mariadb
import Stark_ML
import pandas as pd

from Stark_ML.utils.encoding import *

with open('Stark_ML/credentials.json', 'r') as file:
    creds = json.load(file)


def connect_to_DB(creds):
    conn = mariadb.connect(**creds)
    
    return conn
    
    
def is_valid_element(symbol):
    # List of all valid element symbols in the periodic table
    valid_elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    return symbol.capitalize() in valid_elements
    
    
def convert_species_request(s):
    def parse_roman_part(part):
        part = part.strip()
        if not part:
            return []
        if '-' in part:
            start, end = [p.strip() for p in part.split('-')]
            return list(range(roman.fromRoman(start.upper())-1, roman.fromRoman(end.upper()) + 1))
        return [roman.fromRoman(part.upper())-1]

    def parse_ionization_stages(stages):
        # Separate by commas, then handle each part
        parts = re.split(r'\s*,\s*', stages)
        ionization = []
        for part in parts:
            ionization.extend(parse_roman_part(part))
        return ionization

    elements = []
    ionizations = []
    chem_elems = s.split(';')
    
    for chem_elem in chem_elems:
        match = re.match(r'([A-Za-z]{1,2})\s*([ivx,\s-]*)', chem_elem.strip(), re.IGNORECASE)
        if match:
            element = match.group(1).capitalize()
            if is_valid_element(element) == False:
                raise ValueError(f"Chemical element symbol {element} is incorrect")
            if match.group(2).strip():
                ionization = parse_ionization_stages(match.group(2))
            else:
                ionization = 'All'
            elements.append(element)
            ionizations.append(ionization)
        else:
            raise ValueError("Invalid input format")
    
    return elements, ionizations
    

def get_lines_from_DB(elements: str, lower: str(float), upper: str(float), count_mode = False, save_for_manual_check = False):
        if count_mode:
            lines_count = _handle_query(elements, lower, upper, count_mode)
            return lines_count
        else:
            DB_df = _handle_query(elements, lower, upper, count_mode)
            if DB_df is None:
                raise UserDefinedError('There are no lines of the selected species in this spectral region')
            
            data_i = pd.read_excel(Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/Stark_data.xlsx',
                                   sheet_name='Ions',
                                   usecols='A:BQ',
                                   nrows = 2
                               )
            lines_with_None = DB_df[DB_df.isna().any(axis=1)]
            DB_df           = DB_df[~DB_df.isna().any(axis=1)]
            
            try:
                request_df, lines_for_check = split_OK_check(DB_to_StarkML(DB_df, data_i), save_manual_check = save_for_manual_check, save_txts = False)
                lines_with_None = DB_to_StarkML(lines_with_None, data_i)
                return request_df, pd.concat([lines_for_check, lines_with_None], axis = 0, ignore_index = True)
            except UserDefinedError as e:
                raise UserDefinedError(e)
                
                
def _handle_query(spectra: str,
                    lower: float,
                    upper: float,
                    count_mode: bool):
        elements, ionizations = convert_species_request(spectra)
        
        
        connection = connect_to_DB(creds)
        cur = connection.cursor()
        
        if count_mode:
            response = _get_count_lines_from_DB(cur, elements, ionizations, lower, upper)
        else:
            response = _get_lines_from_DB(cur, elements, ionizations, lower, upper)
        cur.close()
        connection.close()
        return response


def _get_lines_from_DB(cursor, elements, ionizations, lower_wl, upper_wl):
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
        cursor.execute(query, (lower_wl, upper_wl, el))
        column_names = [desc[0] for desc in cursor.description]
        req_results = cursor.fetchall()
        req_results = pd.DataFrame(req_results, columns=column_names)
        if not req_results.empty:
            if DB_df is None:
                DB_df = req_results
            else:
                DB_df = pd.concat([DB_df, req_results], ignore_index=True)
    return DB_df
        
        
def _get_count_lines_from_DB(cursor, elements, ionizations, lower_wl, upper_wl):
    count = 0
    for i in range(len(elements)):
        el = elements[i]
        ion = ionizations[i]
        if ion != 'All':
            query = f'''
            SELECT COUNT(*)
            FROM mytestview2
            WHERE airwl >= ?
            AND airwl <= ?
            AND el_name = ?
            AND ion_stage in {f"({', '.join(map(str, ion))})"}
            '''
        else:
            query = f'''
            SELECT COUNT(*)
            FROM mytestview2
            WHERE airwl >= ?
            AND airwl <= ?
            AND el_name = ?
            '''
        cursor.execute(query, (lower_wl, upper_wl, el))
        req_results = cursor.fetchall()[0][0]
        DB_df = req_results
        count += req_results
    return count