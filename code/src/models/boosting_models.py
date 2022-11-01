import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from ._models import CatBoosting
from ._models import rmse, RMSELoss
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

def objective(trial: Trial, X, y, model_name, model_kind):
    param = {}
    score = None
    if model_name == 'CB':
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
        if model_kind == 'clf':
            model = CatBoostClassifier(**param)
            catboost_model = model.fit(X, y,  early_stopping_rounds=35, verbose=100)
            ## RMSE으로 Loss 계산
            score = rmse( y,catboost_model.predict(X).squeeze(1))
        elif model_kind == 'reg':
            model = CatBoostRegressor(**param)
            catboost_model = model.fit(X, y,  early_stopping_rounds=35, verbose=100)
            ## RMSE으로 Loss 계산
            score = rmse( y,catboost_model.predict(X))
    elif model_name == 'XGB':
        param = {
            # 'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
            # 'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            # 'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            # 'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
            # 'n_estimators': 10000,
            # 'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17]),
            # 'random_state': trial.suggest_categorical('random_state', [2020]),
            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 10),
            "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
            'n_estimators': trial.suggest_int("n_estimators", 400, 4000, 400),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4), # L2 regularization
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4), # L1 regularization
            'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),     
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
            }
        if model_kind == 'clf':
            model = XGBClassifier(**param)
            model.fit(X, y, early_stopping_rounds=100, verbose=100)
            ## RMSE으로 Loss 계산
            score = rmse( y, model.predict(X).squeeze(1))
        elif model_kind == 'reg':
            model = XGBRegressor(**param)
            model.fit(X, y, early_stopping_rounds=100, verbose=100)
            ## RMSE으로 Loss 계산
            score = rmse( y, model.predict(X))
    elif model_name == 'LGBM':
        param = {}
        if model_kind == 'clf':
            model = LGBMClassifier(**param)
            lightgbm_model = model.fit(X, y, early_stopping_rounds=100, verbose=False)
            ## RMSE으로 Loss 계산
            score = rmse( y,lightgbm_model.predict(X).squeeze(1))
        elif model_kind == 'reg':
            model = LGBMRegressor(**param)
            lightgbm_model = model.fit(X, y, early_stopping_rounds=100, verbose=False)
            ## RMSE으로 Loss 계산
            score = rmse( y,lightgbm_model.predict(X))


    return score

class CatBoostingModel:
    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.catmodels = {}
        self.data = data

        # # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
        # study = optuna.create_study(direction='minimize',sampler=TPESampler())      
        # # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
        # study.optimize(lambda trial : objective(trial, self.data['train'].drop('rating',axis=1), self.data['train']['rating']), n_trials=50)
        # print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))        
        
        # self.best_params = study.best_params
        # try:
        #     # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        #     optuna.visualization.plot_param_importances(study)      
        #     # 하이퍼파라미터 최적화 과정을 확인
        #     optuna.visualization.plot_optimization_history(study)
        # except:
        #     pass    

    def train(self):
        print(self.data['folds'])
        for fold in range(8):
            print(f'===================================={fold+1}============================================')
            train_idx, valid_idx = self.data['folds'][fold]
            X_train = self.data['train'].drop(['rating'],axis=1).iloc[train_idx] 
            X_valid = self.data['train'].drop(['rating'],axis=1).iloc[valid_idx]
            y_train = self.data['train']['rating'][train_idx]
            y_valid = self.data['train']['rating'][valid_idx]

            cat = CatBoostRegressor( learning_rate= 0.048, 
                                    depth= 15,
                                    l2_leaf_reg= 1.5,
                                    min_child_samples= 1,
                                    eval_metric = 'RMSE',
                                    bootstrap_type = 'Bernoulli',
                                    iterations = 10000,
                                    grow_policy = 'Depthwise',
                                    use_best_model = True,
                                    od_type = 'Iter',
                                    od_wait = 20,
                                    random_state = 42,
                                    cat_features = self.data['cat_features']
                                    )

            # cat = CatBoostRegressor( **self.best_params,
            #                         random_state = 42
            #                         )
            cat.fit(X_train,y_train, eval_set=(X_valid, y_valid))
            self.catmodels[fold] = cat 
            print(f'================================================================================\n\n')
            #print('epoch:', epoch, 'validation: rmse:', rmse_score)
        # print( **self.best_params) 

    def predict(self):
        preds =np.zeros(self.data['test'].shape[0])  
        for fold in range(8):
            a = self.catmodels[fold].predict(self.data['test']).transpose()
            preds += self.catmodels[fold].predict(self.data['test'])
        print(preds)
        return (preds/8).transpose()
        
class XGBModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.model =  XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5)
        # self.model =  XGBClassifier(n_estimators=2000, learning_rate=0.01, max_depth=5, num_feature=100)
        self.data = data

        # # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
        # study = optuna.create_study(direction='minimize',sampler=TPESampler())

        # # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
        # study.optimize(lambda trial : objective(trial, self.data['X_train'], self.data['y_train'], 'XGB', 'reg'), n_trials=30)
        # # study.optimize(lambda trial : objective(trial, self.data['X_train'], self.data['y_train'], 'XGB', 'clf'), n_trials=30)
        # print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

        # # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        # optuna.visualization.plot_param_importances(study)

        # # 하이퍼파라미터 최적화 과정을 확인
        # optuna.visualization.plot_optimization_history(study)

    def train(self):
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                        self.data['train'].drop(['rating'], axis=1),
                                                        self.data['train']['rating'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True
                                                        )
        self.data['X_train'],  self.data['X_valid'],  self.data['y_train'],  self.data['y_valid'] = X_train, X_valid, y_train, y_valid

        self.model.fit(self.data['X_train'],self.data['y_train'])
        preds = self.model.predict(self.data['X_valid'])
        print('RMSE : ', rmse(self.data['y_valid'], preds)) # Regressor
        # print('RMSE : ', rmse(self.data['y_valid'], preds.squeeze(1))) # Classifier
        #print('epoch:', epoch, 'validation: rmse:', rmse_score)
        return 

    def predict(self):
        preds = self.model.predict(self.data['test'])
        print(preds)
        return preds # Regressor
        # return preds.squeeze(1) # Classifier


class LGBMModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.model =  LGBMRegressor(nthread=4,
                        n_estimators=1000,
                        learning_rate=0.02,
                        num_leaves=34,
                        colsample_bytree=0.94,
                        subsample=0.87,
                        max_depth=8,
                        reg_alpha=0.04,
                        reg_lambda=0.07,
                        min_split_gain=0.02,
                        min_child_weight=32,
                        silent=-1,
                        verbose=-1)
        # self.model =  LGBMClassifier( )
        self.data = data

        # # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
        # study = optuna.create_study(direction='minimize',sampler=TPESampler())

        # # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
        # study.optimize(lambda trial : objective(trial, self.data['X_train'], self.data['y_train'], 'LGBM', 'reg'), n_trials=50)
        # # study.optimize(lambda trial : objective(trial, self.data['X_train'], self.data['y_train'], 'LGBM', 'clf'), n_trials=50)
        # print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

        # # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        # optuna.visualization.plot_param_importances(study)

        # # 하이퍼파라미터 최적화 과정을 확인
        # optuna.visualization.plot_optimization_history(study)

    def train(self):
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                        self.data['train'].drop(['rating'], axis=1),
                                                        self.data['train']['rating'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True
                                                        )
        self.data['X_train'],  self.data['X_valid'],  self.data['y_train'],  self.data['y_valid'] = X_train, X_valid, y_train, y_valid

        self.model.fit(self.data['X_train'],self.data['y_train'])
        preds = self.model.predict(self.data['X_valid'])
        print('RMSE : ', rmse(self.data['y_valid'], preds)) # Regressor
        # print('RMSE : ', rmse(self.data['y_valid'], preds.squeeze(1))) # Classifier
        #print('epoch:', epoch, 'validation: rmse:', rmse_score)
        return 


    def predict(self):
        preds =np.zeros(self.data['test'].shape[0])  
        for fold in range(8):
            a = self.catmodels[fold].predict(self.data['test']).transpose()
            preds += self.catmodels[fold].predict(self.data['test'])
        print(preds)
        return preds # Regressor
        # return preds.squeeze(1) # Classifier
