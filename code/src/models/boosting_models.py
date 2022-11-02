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
from .utils import do_optuna,get_parameter

import json
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'config.json')
params_dict = {}

with open(file_path, 'r') as file:
    params_dict = json.load(file)

class BoostingModel:
    """
    [주의] 모델마다 train, predict 동작이 바뀌어야 하는 경우, class 도 나뉘어야 한다.
    [주의] CatBoostRegressor, XGBoostRegressor, LGBMRegressor 의 경우 train 과 predict 에 별 다른 동작이 없기 때문에 그냥 합쳐 둔다. 
    """
    def __init__(self, args, data):

        super().__init__()

        self.criterion = RMSELoss()
        self.model_name = args.MODEL
        self.data = data

        self.my_model = CatBoostRegressor

        if( "LightGBMRegressor" == self.model_name ):
            print("[INFO] Let's go LGBMRegressor ! ")
            self.my_model = LGBMRegressor
        elif( "XGBRegressor" == self.model_name ): 
            print("[INFO] Let's go XGBRegressor ! ")
            self.my_model = XGBRegressor
        elif( "CatBoostRegressor"== self.model_name ): 
            print("[INFO] Let's go CatBoostRegressor ! ")
            self.my_model = CatBoostRegressor
        else :
            print("[WARN] Use Default Model ") 

        self.best_params = do_optuna( self.data,
                   model = self.my_model,
                   model_name =  args.MODEL,
                   early_stopping_rounds = 35,
                   verbose = 100) if args.DO_OPTUNA == True else params_dict[args.MODEL]
        
        if( "CatBoostRegressor"== self.model_name ): 
            self.best_params['cat_features'] =self.data['cat_features']

    def train(self):

        self.my_model = self.my_model(**self.best_params)
        if( "CatBoostRegressor"== self.model_name ): 
            self.my_model.fit(X_train,y_train, eval_set=(X_valid, y_valid))
        else:
            self.my_model.fit(X_train,y_train)

        self.my_model.fit(self.data['X_train'],self.data['y_train'])
        preds = self.my_model.predict(self.data['X_valid'])
        preds = preds.squeeze(1) if 'Classifier' in self.model_name else preds
        
        print('RMSE : ', rmse(self.data['y_valid'], preds)) # Regressor
        # print('RMSE : ', rmse(self.data['y_valid'], preds.squeeze(1))) # Classifier
        #print('epoch:', epoch, 'validation: rmse:', rmse_score)

    def predict(self):

        preds = self.my_model.predict(self.data['test'])
        return preds

class CatBoostingModel:
    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.catmodels = {}
        self.data = data

        target_model = CatBoostRegressor

        self.best_params = do_optuna( self.data,
                   model = target_model,
                   model_name = 'CatBoostRegressor',
                   early_stopping_rounds = 35,
                   verbose = 100) if args.DO_OPTUNA == True else params_dict['CatBoostRegressor']

        self.best_params['cat_features'] =self.data['cat_features']

    def train(self):
        print(self.data['folds'])
        for fold in range(8):
            print(f'===================================={fold+1}============================================')
            train_idx, valid_idx = self.data['folds'][fold]
            X_train = self.data['train'].drop(['rating'],axis=1).iloc[train_idx] 
            X_valid = self.data['train'].drop(['rating'],axis=1).iloc[valid_idx]
            y_train = self.data['train']['rating'][train_idx]
            y_valid = self.data['train']['rating'][valid_idx]

            cat = CatBoostRegressor( **self.best_params)
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
        self.best_params = do_optuna( self.data,
                        model = self.my_model,
                        model_name =  args.MODEL,
                        early_stopping_rounds = 35,
                        verbose = 100) if args.DO_OPTUNA == True else params_dict[args.MODEL]
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
        self.data = data


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
