from pytorch_tabnet.tab_model import  TabNetRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def bootstrap_metric(x, 
                     y,
                    metric_fn,
                    samples_cnt=1000,
                    alpha=0.05,
                    random_state=777):
    size = len(x)
    
    np.random.seed(random_state)
    b_metric = np.zeros(samples_cnt)
    for it in range(samples_cnt):
        poses = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
        if not isinstance(x, np.ndarray):
          x_boot = x.to_numpy()[poses]
        else:
          x_boot = x[poses]
        y_boot = y[poses]
        
        m_val = metric_fn(x_boot, y_boot)
        b_metric[it] = m_val
            
    return b_metric

def train_and_test_regressor(models, X_train, y_train, X_test, y_test, max_epochs = 200, patience = 20):
  X_train_save, y_train_save = X_train, y_train
  X_test_save, y_test_save = X_test, y_test
  predictions = {}
  for name, model in models.items():
    if isinstance(model, TabNetRegressor):
      X_train, X_test = X_train_save.to_numpy(), X_test_save.to_numpy()
      y_train, y_test = y_train_save.to_numpy().reshape(-1, 1), y_test_save.to_numpy().reshape(-1, 1)
    else:
      X_train, X_test = X_train_save, X_test_save
      y_train, y_test = y_train_save, y_test_save
   
    print(f"Fitting {name}")
    if isinstance(model, TabNet):
      model.fit(X_train, y_train,
                max_epochs = max_epochs,
                eval_set = [(X_test, y_test)],
                eval_name = ['eval'],
                patience = patience
                )
    else:
      model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
      
  boot_scores = {}

  for name, y_pred in predictions.items():
      print(f"Calculating bootstrap score for {name}")
      boot_score = bootstrap_metric(y_test, 
                                      y_pred, 
                                      metric_fn=lambda x, y: mean_squared_error(y_true=x,
                                                                                y_pred=y,
                                                                                squared = False))
      boot_scores[name] = boot_score
      
  
  results = pd.DataFrame(boot_scores)
  # cast to long format
  results = results.melt(value_vars=results.columns,
                      value_name="mse", 
                      var_name="model") 
  return results

