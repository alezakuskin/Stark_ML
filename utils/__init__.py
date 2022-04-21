def name_to_model(model):
    if model == "KNN":
        from Stark_ML.models.base_models import KNN
        return KNN
        
    elif model == "RF":
        from Stark_ML.models.base_models import RandomForest
        return RandomForest
        
    elif model == "XGB":
        from Stark_ML.models.base_models import XGBoost
        return XGBoost
        
    elif model == "GB":
        from Stark_ML.models.base_models import Gradient_Boosting
        return Gradient_Boosting
    
    elif model == "TabNet":
        from Stark_ML.models.tabnet import TabNet
        return TabNet
        
    else:
        raise NotImplementedError(f'Model {model} has not been implemented yet')