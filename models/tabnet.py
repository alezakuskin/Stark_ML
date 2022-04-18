from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np
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
        optimizer = optim.AdamW(self.model.parameters())

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        loss_func = nn.MSELoss()
        y = y.float()
        y_val = y_val.float()
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size = params['batch_size'], shuffle=True,
                                  num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(params['epochs']):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))

                if self.args.objective == "regression":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.model(batch_val_X.to(self.device))

                if self.args.objective == "regression":
                    out = out.squeeze()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + params['early_stopping_rounds'] < epoch:
                print("Validation loss has not improved for %d steps!" % params['early_stopping_rounds'])
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
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

        self.model = TabNetRegressor(**self.params)
        
    def fit(self, X, y, X_val=None, y_val=None):
        X, X_val = X.to_numpy(), X_val.to_numpy()
        y, y_val = y.to_numpy().reshape(-1, 1), y_val.to_numpy().reshape(-1, 1)
        
        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_name=["eval"])
        history = self.model.history
        return history['loss'], history["eval_" + self.metric[0]]

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
        
        return params_out
        
        
    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """ Generate feature attributions for the model input.
            Only strategy are supported: default ("") 
            Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=np.float)
        attributions = self.model.explain(torch.tensor(X, dtype=torch.float32))[0]
        return attributions