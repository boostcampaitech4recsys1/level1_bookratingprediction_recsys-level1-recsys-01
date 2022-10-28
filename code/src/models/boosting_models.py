import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from ._models import CatBoosting
from ._models import rmse, RMSELoss

from catboost import CatBoostRegressor, CatBoostClassifier, Pool

import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

def objective(trial: Trial, X, y):
    param = {
      "random_state":42,
      'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
      'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
      "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
      "max_depth":trial.suggest_int("max_depth", 4, 16),
      'random_strength' :trial.suggest_int('random_strength', 0, 100),
      "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "max_bin": trial.suggest_int("max_bin", 200, 500),
      'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
    }
  
    model = CatBoostClassifier(**param)
    catboost_model = model.fit(X, y,  early_stopping_rounds=35,verbose=100)

    ## RMSE으로 Loss 계산
    score = rmse( y,catboost_model.predict(X).squeeze(1))

    return score

class CatBoostingModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.model =  CatBoostClassifier( learning_rate = 0.22711164423706456, bagging_temperature= 16.06572911754189, n_estimators= 2879, max_depth= 11, random_strength= 1, colsample_bylevel= 0.8343464926823253, l2_leaf_reg=7.695579834959398e-06, min_child_samples= 8, max_bin= 451, od_type= 'IncToDec')
        self.data = data

#         # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
#         study = optuna.create_study(direction='minimize',sampler=TPESampler())

#         # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
#         study.optimize(lambda trial : objective(trial, self.data['X_train'], self.data['y_train']), n_trials=50)
#         print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

#         # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
#         optuna.visualization.plot_param_importances(study)

#         # 하이퍼파라미터 최적화 과정을 확인
#         optuna.visualization.plot_optimization_history(study)

    def train(self):
        self.model.fit(self.data['X_train'],self.data['y_train'])
        preds = self.model.predict(self.data['X_valid'])
        print('RMSE : ', rmse(self.data['y_valid'], preds.squeeze(1)))
        #print('epoch:', epoch, 'validation: rmse:', rmse_score)
        return 

    def predict(self):
        preds = self.model.predict(self.data['test'])
        print(preds)
        return preds.squeeze(1)
