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
        
        score = 0
        # Cross validate the chosen hyperparameters

        kf = KFold(self.params['nfold'], shuffle = False)
        for train, test in kf.split(self.X):
            model = self.model_name(trial_params)
            model.fit(self.X.iloc[train, :], self.y.iloc[train])
            score += mean_squared_error(self.y.iloc[test], model.predict(self.X.iloc[test, :]),
                                        squared = self.params['squared_metrics'])

        score /= self.params['nfold']

        #Sklearn cross_val_score doesn't work because it need sklearn model
        '''score = -cross_val_score(estimator = model,
                                X = self.X, y = self.y,
                                cv = params['nfold'],
                                scoring = 'neg_root_mean_squared_error')'''

        return score


def main(X, y, model_name, params, n_trials = 100):
    print("Start hyperparameter optimization")
    
    Sampler = optuna.samplers.TPESampler(seed = 777)
    study = optuna.create_study(sampler = Sampler)
    study.optimize(Objective(model_name, X, y, params), n_trials, show_progress_bar = True, n_jobs = -1)
    print("Best parameters:", study.best_trial.params)

    return study