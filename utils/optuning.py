import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

class Objective(object):
    def __init__(self, model_name, X, y, params):
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y
        self.params = params

        
    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.params)
        print(trial_params)

        # Create model
        model = self.model_name(trial_params)

        #model.fit(self.X, self.y)
        score = 0
        # Cross validate the chosen hyperparameters

        kf = KFold(self.params['nfold'])
        for train, test in kf.split(self.X):
            model.fit(self.X.iloc[train, :], self.y.iloc[train])
            score += mean_squared_error(self.y.iloc[test], model.predict(self.X.iloc[test, :]), squared = True)

        score /= self.params['nfold']

        #Sklearn cross_val_score doesn't work because it need sklearn model
        '''score = -cross_val_score(estimator = model,
                                X = self.X, y = self.y,
                                cv = params['nfold'],
                                scoring = 'neg_root_mean_squared_error')'''

        return score


def main(X, y, model_name, params, n_trials = 100):
    print("Start hyperparameter optimization")
    
    study = optuna.create_study()
    study.optimize(Objective(model_name, X, y, params), n_trials)
    print("Best parameters:", study.best_trial.params)

    return study

