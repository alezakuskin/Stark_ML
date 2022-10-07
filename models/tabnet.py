from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from Stark_ML.models.base_models import BaseModel


class BaseModelTorch(BaseModel):

    def __init__(self, params):
        super().__init__(params)
        self.device = self.get_device()
        #self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None

    def to_device(self):
        #if self.args.data_parallel:
        #    self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        #print("On Device:", self.device)
        self.model.to(self.device)
        
    def get_device(self):
        if torch.cuda.is_available():
                device = "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
        else:
            device = 'cpu'

        return torch.device(device)

    def fit(self, X, y, params, X_val=None, y_val=None):
        return loss_history, val_loss_history

    def predict(self, X):
        self.predictions = self.predict_helper(X)
        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X, params):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False,
                                 num_workers=2)
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")

        
class TabNet(BaseModelTorch):

    def __init__(self, params):
        super().__init__(params)

        # Paper recommends to be n_d and n_a the same
        '''self.params["n_a"] = self.params["n_d"]

        self.params["cat_idxs"] = args.cat_idx
        self.params["cat_dims"] = args.cat_dims

        self.params["device_name"] = self.device'''

        self.model = TabNetRegressor(**params, n_a = params['n_d'], verbose = False, seed = np.random.randint(1000))
        
    def fit(self, X, y, X_val=None, y_val=None):
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1, 1)
        
        if isinstance(X_val, pd.DataFrame):
            X_val, y_val = X_val.to_numpy(), y_val.to_numpy().reshape(-1, 1)
            
        self.model.fit(X, y, eval_set = [(X_val, y_val)], eval_name = ['eval'], max_epochs = 500, patience = 20)
        history = self.model.history
        return history['loss']

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float)

        return self.model.predict(X)
        
    @classmethod
    def define_trial_parameters(cls, trial, params):
        params_tunable = {}
        params_out = {}
        for i, val in params.items():
            if isinstance(val, list):
                params_tunable[f'{i}'] = val
            else:
                params_out[f'{i}'] = val
        
        if 'n_d' in params_tunable:
            params_out[f'n_d'] = trial.suggest_int('n_d', params['n_d'][0], params['n_d'][1], log = False)
        if 'n_steps' in params_tunable:
            params_out[f'n_steps'] = trial.suggest_int('n_steps', params['n_steps'][0], params['n_steps'][1], log = False)
        if 'gamma' in params_tunable:
            params_out[f'gamma'] = trial.suggest_float('gamma', params['gamma'][0], params['gamma'][1], log = False)
        if 'cat_emb_dim' in params_tunable:
            params_out[f'cat_emb_dim'] = trial.suggest_int('cat_emb_dim', params['cat_emb_dim'][0], params['cat_emb_dim'][1], log = False)
        if 'n_independent' in params_tunable:
            params_out[f'n_independent'] = trial.suggest_int('n_independent', params['n_independent'][0], params['n_independent'][1], log = False)
        if 'n_shared' in params_tunable:
            params_out[f'n_shared'] = trial.suggest_int('n_shared', params['n_shared'][0], params['n_shared'][1], log = False)
        if 'momentum' in params_tunable:
            params_out[f'momentum'] = trial.suggest_float('momentum', params['momentum'][0], params['momentum'][1], log = True)
        if 'mask_type' in params_tunable:
            params_out[f'mask_type'] = trial.suggest_categorical('mask_type', params['mask_type'])
        
        
        if 'nfold' in params_out:
            del params_out['nfold']
        if 'squared_metrics' in params_out:
            del params_out['squared_metrics']
        if 'n_jobs' in params_out:
            del params_out['n_jobs']
        
        return params_out
        
        
    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """ Generate feature attributions for the model input.
            Only strategy are supported: default ("") 
            Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=np.float)
        attributions = self.model.explain(torch.tensor(X, dtype=torch.float32))[0]
        return attributions