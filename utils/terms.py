import Stark_ML
import pandas as pd

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
    else:
      raise NameError(f'Term symbol "{val}" is not specified')
  return momentum



def gap_to_ion(data, column_name = None, file = Stark_ML.__path__.__dict__['_path'][0] + '/Source_files/E_ion.csv'):
    import pandas as pd
    import numpy as np
    ion_Es = pd.read_csv(file)
    gap = pd.Series()
    for index, val in enumerate(data['Gap to ion']):
        gap.at[index] = float(ion_Es.loc[ion_Es['Element'] == data.loc[index]['Element']][str(data.loc[index]['Charge'])]) - data.loc[index][column_name]
        if np.isnan(gap.at[index]):
            print(f"Please find and insert to '/Source_files/E_ion.csv' ionization energy value for {data.loc[index]['Element']} with charge {data.loc[index]['Charge']}")
    return gap