
def test_selection(data, target, p = 0.2, low_limit = 'min', up_limit = 'max', random_state = None):
  if low_limit != 'min':
    low_limit = str(low_limit) + '%'
  if up_limit != 'max':
    up_limit = str(up_limit) + '%'
  
  stats = target.describe()
  target_range = target[(target >= stats[low_limit]) & (target <= stats[up_limit])]
  target_selected = target_range.sample(frac = p, random_state = random_state)
  data_selected = data.loc[target_selected.index.to_list()]

  return data_selected, target_selected

