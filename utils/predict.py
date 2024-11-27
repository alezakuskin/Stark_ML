import joblib
import xgboost
import catboost
from Stark_ML.utils.encoding import *

class Predictor:

    def __init__(
        self,
        path_models = Stark_ML.__path__.__dict__['_path'][0]
    ):
        """
        Loads pretrained models and scalers for width and shift prediction
        
        Parameters
        ----------
        path_models (str): path to folder with saved pretrained models"""
        
        #Import pretrained models for width prediction
        self.model1 = xgboost.XGBRegressor()
        self.model1.load_model(path_models + '/XGB_A+I_Eraw_Raw_No.json')

        self.model2 = xgboost.XGBRegressor()
        self.model2.load_model(path_models + '/XGB_A+I_Enorm_Aug_No.json')

        self.model3 = catboost.CatBoostRegressor()
        self.model3.load_model(path_models + '/CatBoost_A+I_Enorm_Raw_No.json')

        self.model4 = joblib.load(path_models + '/LightGBM_A+I_Eraw_Raw_No.pkl')

        self.model5 = joblib.load(path_models + '/LightGBM_A+I_Enorm_Raw_Scaler.pkl')
        
        #Import pretrained model for shift prediction
        self.model_both = joblib.load(path_models + '/RF_Both_Eraw_Aug_No.pkl')

        #Import Standard Scaler
        self.scaler_width = joblib.load(path_models + '/scaler_width.pkl')
        #self.scaler_shift = joblib.load(path_models + '/scaler_shift.pkl')
        
        self.epsilon = 1e-3
        
        
    def predict_width(self, data_for_prediction: pd.DataFrame) -> np.ndarray:
        '''Get predicted Stark broadening parameters for input lines
        
        Parameters
        ----------
        data_for_prediction : pd.DataFrame, dataframe with any number of rows,
            all values of input features filled in; with "Element", "Wavelength",
            "Z number", "w (A)", "d (A)" columns.
        
        Returns
        ----------
        numpy.ndarray
            A one-dimentional array with predicted values of broadening parameters in \u212B
        '''
        #Models without energy normalization
        pred1 = self.model1.predict(data_for_prediction.drop(columns=['Element', 'Wavelength', 'Z number']))
        pred4 = self.model4.predict(data_for_prediction.drop(columns=['Element', 'Wavelength', 'Z number']))
        #Models with energy normalization
        data_for_prediction['E lower']    = energy_to_fraction(data_for_prediction, 'E lower')
        data_for_prediction['E upper']    = energy_to_fraction(data_for_prediction, 'E upper')
        data_for_prediction['Gap to ion'] = energy_to_fraction(data_for_prediction, 'Gap to ion')
        pred2 = self.model2.predict(data_for_prediction.drop(columns=['Element', 'Wavelength', 'Z number']))
        pred3 = self.model3.predict(data_for_prediction.drop(columns=['Element', 'Wavelength', 'Z number']))
        pred5 = self.model5.predict(self.scaler_width.transform(data_for_prediction.drop(columns=['Element', 'Wavelength', 'Z number'])))
        preds = (pred1 + pred2 + pred3 + pred4 + pred5)/5
        preds = (np.exp(preds) - 1) * self.epsilon
        
        return preds
        
    def predict_shift(self, data_for_prediction: pd.DataFrame) -> np.ndarray:
        '''Get predicted Stark shift parameters for input lines
        
        Parameters
        ----------
        data_for_prediction : pd.DataFrame, dataframe with any number of rows,
            all values of input features filled in; with "Element", "Wavelength",
            "Z number", "w (A)", "d (A)" columns.
        
        Returns
        ----------
        numpy.ndarray
            A two-dimentional array with predicted values of both broadening (1-st column)
            and shift (2nd column) parameters in \u212B
        '''
        #Get broadening predictions first
        widths = self.predict_width(data_for_prediction)
        
        #Adjust input data
        data_for_prediction['w (A)'] = widths
        data_for_prediction = data_for_prediction[self.model_both.model.feature_names_in_]
        
        #Get shift predictions
        preds = self.model_both.predict(data_for_prediction)
        
        return np.column_stack((widths, preds))