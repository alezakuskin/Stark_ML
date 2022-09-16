def term_to_number(term):
  import pandas as pd
  momentum = pd.Series()
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

